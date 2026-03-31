# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import socket
import unittest
from io import BytesIO
import time
from threading import Lock
from unittest.mock import Mock, ANY, call, patch

from cassandra import OperationTimedOut
from cassandra.cluster import Cluster
from cassandra.connection import (Connection, HEADER_DIRECTION_TO_CLIENT, ProtocolError,
                                  locally_supported_compressions, ConnectionHeartbeat, _Frame, Timer, TimerManager,
                                  ConnectionException, ConnectionShutdown, DefaultEndPoint, ShardAwarePortGenerator)
from cassandra.marshal import uint8_pack, uint32_pack, int32_pack
from cassandra.protocol import (write_stringmultimap, write_int, write_string,
                                SupportedMessage, ProtocolHandler, ResultMessage,
                                RESULT_KIND_SET_KEYSPACE)

from tests.util import wait_until, assertRegex
import pytest


class ConnectionTest(unittest.TestCase):

    def make_connection(self, **kwargs):
        c = Connection(DefaultEndPoint('1.2.3.4'), **kwargs)
        c._socket = Mock()
        c._socket.send.side_effect = lambda x: len(x)
        return c

    def make_header_prefix(self, message_class, version=Connection.protocol_version, stream_id=0):
        return bytes().join(map(uint8_pack, [
            0xff & (HEADER_DIRECTION_TO_CLIENT | version),
            0,  # flags (compression)
            0,  # MSB for v3+ stream
            stream_id,
            message_class.opcode  # opcode
        ]))

    def make_options_body(self):
        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['3.0.1'],
            'COMPRESSION': []
        })
        return options_buf.getvalue()

    def make_error_body(self, code, msg):
        buf = BytesIO()
        write_int(buf, code)
        write_string(buf, msg)
        return buf.getvalue()

    def make_msg(self, header, body=""):
        return header + uint32_pack(len(body)) + body

    def test_connection_endpoint(self):
        endpoint = DefaultEndPoint('1.2.3.4')
        c = Connection(endpoint)
        assert c.endpoint == endpoint
        assert c.endpoint.address == endpoint.address

        c = Connection(host=endpoint)  # kwarg
        assert c.endpoint == endpoint
        assert c.endpoint.address == endpoint.address

        c = Connection('10.0.0.1')
        endpoint = DefaultEndPoint('10.0.0.1')
        assert c.endpoint == endpoint
        assert c.endpoint.address == endpoint.address

    def test_bad_protocol_version(self, *args):
        c = self.make_connection()
        c._requests = Mock()
        c.defunct = Mock()

        # read in a SupportedMessage response
        header = self.make_header_prefix(SupportedMessage, version=0x7f)
        options = self.make_options_body()
        message = self.make_msg(header, options)
        c._iobuf._io_buffer = BytesIO()
        c._iobuf.write(message)
        c.process_io_buffer()

        # make sure it errored correctly
        c.defunct.assert_called_once_with(ANY)
        args, kwargs = c.defunct.call_args
        assert isinstance(args[0], ProtocolError)

    def test_negative_body_length(self, *args):
        c = self.make_connection()
        c._requests = Mock()
        c.defunct = Mock()

        # read in a SupportedMessage response
        header = self.make_header_prefix(SupportedMessage)
        message = header + int32_pack(-13)
        c._iobuf._io_buffer = BytesIO()
        c._iobuf.write(message)
        c.process_io_buffer()

        # make sure it errored correctly
        c.defunct.assert_called_once_with(ANY)
        args, kwargs = c.defunct.call_args
        assert isinstance(args[0], ProtocolError)

    def test_unsupported_cql_version(self, *args):
        c = self.make_connection()
        c._requests = {0: (c._handle_options_response, ProtocolHandler.decode_message, [])}
        c.defunct = Mock()
        c.cql_version = "3.0.3"

        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['7.8.9'],
            'COMPRESSION': []
        })
        options = options_buf.getvalue()

        c.process_msg(_Frame(version=4, flags=0, stream=0, opcode=SupportedMessage.opcode, body_offset=9, end_pos=9 + len(options)), options)

        # make sure it errored correctly
        c.defunct.assert_called_once_with(ANY)
        args, kwargs = c.defunct.call_args
        assert isinstance(args[0], ProtocolError)

    def test_prefer_lz4_compression(self, *args):
        c = self.make_connection()
        c._requests = {0: (c._handle_options_response, ProtocolHandler.decode_message, [])}
        c.defunct = Mock()
        c.cql_version = "3.0.3"

        locally_supported_compressions.pop('lz4', None)
        locally_supported_compressions.pop('snappy', None)
        locally_supported_compressions['lz4'] = ('lz4compress', 'lz4decompress')
        locally_supported_compressions['snappy'] = ('snappycompress', 'snappydecompress')

        # read in a SupportedMessage response
        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['3.0.3'],
            'COMPRESSION': ['snappy', 'lz4']
        })
        options = options_buf.getvalue()

        c.process_msg(_Frame(version=4, flags=0, stream=0, opcode=SupportedMessage.opcode, body_offset=9, end_pos=9 + len(options)), options)

        assert c.decompressor == locally_supported_compressions['lz4'][1]

    def test_requested_compression_not_available(self, *args):
        c = self.make_connection()
        c._requests = {0: (c._handle_options_response, ProtocolHandler.decode_message, [])}
        c.defunct = Mock()
        # request lz4 compression
        c.compression = "lz4"

        locally_supported_compressions.pop('lz4', None)
        locally_supported_compressions.pop('snappy', None)
        locally_supported_compressions['lz4'] = ('lz4compress', 'lz4decompress')
        locally_supported_compressions['snappy'] = ('snappycompress', 'snappydecompress')

        # the server only supports snappy
        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['3.0.3'],
            'COMPRESSION': ['snappy']
        })
        options = options_buf.getvalue()

        c.process_msg(_Frame(version=4, flags=0, stream=0, opcode=SupportedMessage.opcode, body_offset=9, end_pos=9 + len(options)), options)

        # make sure it errored correctly
        c.defunct.assert_called_once_with(ANY)
        args, kwargs = c.defunct.call_args
        assert isinstance(args[0], ProtocolError)

    def test_use_requested_compression(self, *args):
        c = self.make_connection(protocol_version=4)
        c._requests = {0: (c._handle_options_response, ProtocolHandler.decode_message, [])}
        c.defunct = Mock()
        # request snappy compression
        c.compression = "snappy"

        locally_supported_compressions.pop('lz4', None)
        locally_supported_compressions.pop('snappy', None)
        locally_supported_compressions['lz4'] = ('lz4compress', 'lz4decompress')
        locally_supported_compressions['snappy'] = ('snappycompress', 'snappydecompress')

        # the server only supports snappy
        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['3.0.3'],
            'COMPRESSION': ['snappy', 'lz4']
        })
        options = options_buf.getvalue()

        c.process_msg(_Frame(version=4, flags=0, stream=0, opcode=SupportedMessage.opcode, body_offset=9, end_pos=9 + len(options)), options)

        assert c.decompressor == locally_supported_compressions['snappy'][1]

    def test_disable_compression(self, *args):
        c = self.make_connection()
        c._requests = {0: (c._handle_options_response, ProtocolHandler.decode_message)}
        c.defunct = Mock()
        # disable compression
        c.compression = False

        locally_supported_compressions.pop('lz4', None)
        locally_supported_compressions.pop('snappy', None)
        locally_supported_compressions['lz4'] = ('lz4compress', 'lz4decompress')
        locally_supported_compressions['snappy'] = ('snappycompress', 'snappydecompress')

        # read in a SupportedMessage response
        header = self.make_header_prefix(SupportedMessage)

        # the server only supports snappy
        options_buf = BytesIO()
        write_stringmultimap(options_buf, {
            'CQL_VERSION': ['3.0.3'],
            'COMPRESSION': ['snappy', 'lz4']
        })
        options = options_buf.getvalue()

        message = self.make_msg(header, options)
        c.process_msg(message, len(message) - 8)

        assert c.decompressor == None

    def test_not_implemented(self):
        """
        Ensure the following methods throw NIE's. If not, come back and test them.
        """
        c = self.make_connection()
        with pytest.raises(NotImplementedError):
            c.close()

    def test_set_keyspace_blocking(self):
        c = self.make_connection()

        assert c.keyspace == None
        c.set_keyspace_blocking(None)
        assert c.keyspace == None

        c.keyspace = 'ks'
        c.set_keyspace_blocking('ks')
        assert c.keyspace == 'ks'

    def test_set_keyspace_blocking_escapes_quotes(self):
        """
        Test that set_keyspace_blocking properly escapes double quotes in
        keyspace names to prevent CQL injection. This is the Python equivalent
        of the vulnerability fixed in the Go driver:
        https://github.com/scylladb/gocql/pull/783
        """
        c = self.make_connection()
        c.wait_for_response = Mock(return_value=ResultMessage(kind=RESULT_KIND_SET_KEYSPACE))

        c.set_keyspace_blocking('my"ks')
        query_msg = c.wait_for_response.call_args[0][0]
        assert query_msg.query == 'USE "my""ks"', (
            "Double quotes in keyspace name must be escaped as double-double quotes")

    def test_set_keyspace_async_escapes_quotes(self):
        """
        Test that set_keyspace_async properly escapes double quotes in
        keyspace names to prevent CQL injection.
        """
        c = self.make_connection()
        c.lock = Lock()
        c.in_flight = 0
        c.max_request_id = 100
        c.get_request_id = Mock(return_value=1)
        c.send_msg = Mock()

        callback = Mock()
        c.set_keyspace_async('my"ks', callback)

        query_msg = c.send_msg.call_args[0][0]
        assert query_msg.query == 'USE "my""ks"', (
            "Double quotes in keyspace name must be escaped as double-double quotes")

    def test_set_connection_class(self):
        cluster = Cluster(connection_class='test')
        assert 'test' == cluster.connection_class

    def test_connection_shutdown_includes_last_error(self):
        """
        Test that ConnectionShutdown exceptions include the last_error when available.
        This helps debug issues like "Bad file descriptor" by showing the original cause.
        See https://github.com/scylladb/python-driver/issues/614
        """
        c = self.make_connection()
        c.lock = Lock()
        c._requests = {}

        # Simulate the connection becoming defunct with a specific error
        original_error = OSError(9, "Bad file descriptor")
        c.is_defunct = True
        c.last_error = original_error

        # send_msg should raise ConnectionShutdown that includes the last_error
        with pytest.raises(ConnectionShutdown) as exc_info:
            c.send_msg(Mock(), 1, Mock())

        # Verify the error message includes the original error
        error_message = str(exc_info.value)
        assert "is defunct" in error_message
        assert "Bad file descriptor" in error_message

    def test_connection_shutdown_closed_includes_last_error(self):
        """
        Test that ConnectionShutdown exceptions for closed connections include last_error.
        """
        c = self.make_connection()
        c.lock = Lock()
        c._requests = {}

        # Simulate the connection being closed with a specific error
        original_error = OSError(9, "Bad file descriptor")
        c.is_closed = True
        c.last_error = original_error

        # send_msg should raise ConnectionShutdown that includes the last_error
        with pytest.raises(ConnectionShutdown) as exc_info:
            c.send_msg(Mock(), 1, Mock())

        # Verify the error message includes the original error
        error_message = str(exc_info.value)
        assert "is closed" in error_message
        assert "Bad file descriptor" in error_message

    def test_wait_for_responses_shutdown_includes_last_error(self):
        """
        Test that wait_for_responses raises ConnectionShutdown with last_error.
        """
        c = self.make_connection()
        c.lock = Lock()
        c._requests = {}

        # Simulate the connection being defunct with a specific error
        original_error = OSError(9, "Bad file descriptor")
        c.is_defunct = True
        c.last_error = original_error

        # wait_for_responses should raise ConnectionShutdown that includes the last_error
        with pytest.raises(ConnectionShutdown) as exc_info:
            c.wait_for_responses(Mock())

        # Verify the error message includes the original error
        error_message = str(exc_info.value)
        assert "already closed" in error_message
        assert "Bad file descriptor" in error_message


@patch('cassandra.connection.ConnectionHeartbeat._raise_if_stopped')
class ConnectionHeartbeatTest(unittest.TestCase):

    @staticmethod
    def make_get_holders(len):
        holders = []
        for _ in range(len):
            holder = Mock()
            holder.get_connections = Mock(return_value=[])
            holders.append(holder)
        get_holders = Mock(return_value=holders)
        return get_holders

    def run_heartbeat(self, get_holders_fun, count=2, interval=0.05, timeout=0.05):
        ch = ConnectionHeartbeat(interval, get_holders_fun, timeout=timeout)
        # wait until the thread is started
        wait_until(lambda: get_holders_fun.call_count > 0, 0.01, 100)
        time.sleep(interval * (count-1))
        ch.stop()
        assert get_holders_fun.call_count

    def test_empty_connections(self, *args):
        count = 3
        get_holders = self.make_get_holders(1)

        self.run_heartbeat(get_holders, count)

        assert get_holders.call_count >= count-1
        assert get_holders.call_count <= count
        holder = get_holders.return_value[0]
        holder.get_connections.assert_has_calls([call()] * get_holders.call_count)

    def test_idle_non_idle(self, *args):
        request_id = 999

        # connection.send_msg(OptionsMessage(), connection.get_request_id(), self._options_callback)
        def send_msg(msg, req_id, msg_callback):
            msg_callback(SupportedMessage([], {}))

        idle_connection = Mock(spec=Connection, host='localhost',
                               max_request_id=127,
                               lock=Lock(),
                               in_flight=0, is_idle=True,
                               is_defunct=False, is_closed=False,
                               get_request_id=lambda: request_id,
                               send_msg=Mock(side_effect=send_msg))
        non_idle_connection = Mock(spec=Connection, in_flight=0, is_idle=False, is_defunct=False, is_closed=False)

        get_holders = self.make_get_holders(1)
        holder = get_holders.return_value[0]
        holder.get_connections.return_value.append(idle_connection)
        holder.get_connections.return_value.append(non_idle_connection)

        self.run_heartbeat(get_holders)

        holder.get_connections.assert_has_calls([call()] * get_holders.call_count)
        assert idle_connection.in_flight == 0
        assert non_idle_connection.in_flight == 0

        idle_connection.send_msg.assert_has_calls([call(ANY, request_id, ANY)] * get_holders.call_count)
        assert non_idle_connection.send_msg.call_count == 0

    def test_closed_defunct(self, *args):
        get_holders = self.make_get_holders(1)
        closed_connection = Mock(spec=Connection, in_flight=0, is_idle=False, is_defunct=False, is_closed=True)
        defunct_connection = Mock(spec=Connection, in_flight=0, is_idle=False, is_defunct=True, is_closed=False)
        holder = get_holders.return_value[0]
        holder.get_connections.return_value.append(closed_connection)
        holder.get_connections.return_value.append(defunct_connection)

        self.run_heartbeat(get_holders)

        holder.get_connections.assert_has_calls([call()] * get_holders.call_count)
        assert closed_connection.in_flight == 0
        assert defunct_connection.in_flight == 0
        assert closed_connection.send_msg.call_count == 0
        assert defunct_connection.send_msg.call_count == 0

    def test_no_req_ids(self, *args):
        in_flight = 3

        get_holders = self.make_get_holders(1)
        max_connection = Mock(spec=Connection, host='localhost',
                              lock=Lock(),
                              max_request_id=in_flight - 1, in_flight=in_flight,
                              is_idle=True, is_defunct=False, is_closed=False)
        holder = get_holders.return_value[0]
        holder.get_connections.return_value.append(max_connection)

        self.run_heartbeat(get_holders)

        holder.get_connections.assert_has_calls([call()] * get_holders.call_count)
        assert max_connection.in_flight == in_flight
        assert max_connection.send_msg.call_count == 0
        assert max_connection.send_msg.call_count == 0
        max_connection.defunct.assert_has_calls([call(ANY)] * get_holders.call_count)
        holder.return_connection.assert_has_calls(
            [call(max_connection)] * get_holders.call_count)

    def test_unexpected_response(self, *args):
        request_id = 999

        get_holders = self.make_get_holders(1)

        def send_msg(msg, req_id, msg_callback):
            msg_callback(object())

        connection = Mock(spec=Connection, host='localhost',
                          max_request_id=127,
                          lock=Lock(),
                          in_flight=0, is_idle=True,
                          is_defunct=False, is_closed=False,
                          get_request_id=lambda: request_id,
                          send_msg=Mock(side_effect=send_msg))
        holder = get_holders.return_value[0]
        holder.get_connections.return_value.append(connection)

        self.run_heartbeat(get_holders)

        assert connection.in_flight == get_holders.call_count
        connection.send_msg.assert_has_calls([call(ANY, request_id, ANY)] * get_holders.call_count)
        connection.defunct.assert_has_calls([call(ANY)] * get_holders.call_count)
        exc = connection.defunct.call_args_list[0][0][0]
        assert isinstance(exc, ConnectionException)
        assertRegex(exc.args[0], r'^Received unexpected response to OptionsMessage.*')
        holder.return_connection.assert_has_calls(
            [call(connection)] * get_holders.call_count)

    def test_timeout(self, *args):
        request_id = 999

        get_holders = self.make_get_holders(1)

        def send_msg(msg, req_id, msg_callback):
            pass

        # we used endpoint=X here because it's a mock and we need connection.endpoint to be set
        connection = Mock(spec=Connection, endpoint=DefaultEndPoint('localhost'),
                          max_request_id=127,
                          lock=Lock(),
                          in_flight=0, is_idle=True,
                          is_defunct=False, is_closed=False,
                          get_request_id=lambda: request_id,
                          send_msg=Mock(side_effect=send_msg))
        holder = get_holders.return_value[0]
        holder.get_connections.return_value.append(connection)

        self.run_heartbeat(get_holders)

        assert connection.in_flight == get_holders.call_count
        connection.send_msg.assert_has_calls([call(ANY, request_id, ANY)] * get_holders.call_count)
        connection.defunct.assert_has_calls([call(ANY)] * get_holders.call_count)
        exc = connection.defunct.call_args_list[0][0][0]
        assert isinstance(exc, OperationTimedOut)
        assert exc.errors == 'Connection heartbeat timeout after 0.05 seconds'
        assert exc.last_host == DefaultEndPoint('localhost')
        holder.return_connection.assert_has_calls(
            [call(connection)] * get_holders.call_count)


class TimerTest(unittest.TestCase):

    def test_timer_collision(self):
        # simple test demonstrating #466
        # same timeout, comparison will defer to the Timer object itself
        t1 = Timer(0, lambda: None)
        t2 = Timer(0, lambda: None)
        t2.end = t1.end

        tm = TimerManager()
        tm.add_timer(t1)
        tm.add_timer(t2)
        # Prior to #466: "TypeError: unorderable types: Timer() < Timer()"
        tm.service_timeouts()


class DefaultEndPointTest(unittest.TestCase):

    def test_default_endpoint_properties(self):
        endpoint = DefaultEndPoint('10.0.0.1')
        assert endpoint.address == '10.0.0.1'
        assert endpoint.port == 9042
        assert str(endpoint) == '10.0.0.1:9042'

        endpoint = DefaultEndPoint('10.0.0.1', 8888)
        assert endpoint.address == '10.0.0.1'
        assert endpoint.port == 8888
        assert str(endpoint) == '10.0.0.1:8888'

    def test_endpoint_equality(self):
        assert DefaultEndPoint('10.0.0.1') == DefaultEndPoint('10.0.0.1')

        assert DefaultEndPoint('10.0.0.1') == DefaultEndPoint('10.0.0.1', 9042)

        assert DefaultEndPoint('10.0.0.1') != DefaultEndPoint('10.0.0.2')

        assert DefaultEndPoint('10.0.0.1') != DefaultEndPoint('10.0.0.1', 0000)

    def test_endpoint_resolve(self):
        assert DefaultEndPoint('10.0.0.1').resolve() == ('10.0.0.1', 9042)

        assert DefaultEndPoint('10.0.0.1', 3232).resolve() == ('10.0.0.1', 3232)


class TestShardawarePortGenerator(unittest.TestCase):
    @patch('random.randrange')
    def test_generate_ports_basic(self, mock_randrange):
        mock_randrange.return_value = 10005
        gen = ShardAwarePortGenerator(10000, 10020)
        ports = list(itertools.islice(gen.generate(shard_id=1, total_shards=3), 5))

        # Starting from aligned 10005 + shard_id (1), step by 3
        assert ports == [10006, 10009, 10012, 10015, 10018]

    @patch('random.randrange')
    def test_wraps_around_to_start(self, mock_randrange):
        mock_randrange.return_value = 10008
        gen = ShardAwarePortGenerator(10000, 10020)
        ports = list(itertools.islice(gen.generate(shard_id=2, total_shards=4), 5))

        # Expected wrap-around from start_port after end_port is exceeded
        assert ports == [10010, 10014, 10018, 10002, 10006]

    @patch('random.randrange')
    def test_all_ports_have_correct_modulo(self, mock_randrange):
        mock_randrange.return_value = 10012
        total_shards = 5
        shard_id = 3
        gen = ShardAwarePortGenerator(10000, 10020)

        for port in gen.generate(shard_id=shard_id, total_shards=total_shards):
            assert port % total_shards == shard_id

    @patch('random.randrange')
    def test_generate_is_repeatable_with_same_mock(self, mock_randrange):
        mock_randrange.return_value = 10010
        gen = ShardAwarePortGenerator(10000, 10020)

        first_run = list(itertools.islice(gen.generate(0, 2), 5))
        second_run = list(itertools.islice(gen.generate(0, 2), 5))

        assert first_run == second_run


class TestSocketOptions(unittest.TestCase):
    """Unit tests for Connection._apply_socket_options()."""

    def _make_conn(self, **overrides):
        """Create a Connection without actually connecting."""
        conn = object.__new__(Connection)
        # Set defaults that _apply_socket_options expects
        conn.sockopts = None
        conn.tcp_fastopen_connect = False
        conn.tcp_keepalive = False
        conn.tcp_keepalive_idle = 30
        conn.tcp_keepalive_interval = 5
        conn.tcp_keepalive_count = 6
        conn.tcp_user_timeout = None
        for k, v in overrides.items():
            setattr(conn, k, v)
        conn._socket = Mock()
        return conn

    # --- TCP Fast Open ---

    @patch.object(socket, 'TCP_FASTOPEN_CONNECT', 30, create=True)
    def test_tfo_enabled(self):
        conn = self._make_conn(tcp_fastopen_connect=True)
        conn._apply_socket_options(socket.AF_INET)
        conn._socket.setsockopt.assert_any_call(
            socket.IPPROTO_TCP, socket.TCP_FASTOPEN_CONNECT, 1)

    @patch.object(socket, 'TCP_FASTOPEN_CONNECT', 30, create=True)
    def test_tfo_disabled(self):
        conn = self._make_conn(tcp_fastopen_connect=False)
        conn._apply_socket_options(socket.AF_INET)
        calls = conn._socket.setsockopt.call_args_list
        for c in calls:
            self.assertNotEqual(c[0][1], socket.TCP_FASTOPEN_CONNECT)

    def test_tfo_missing_constant(self):
        """No error when TCP_FASTOPEN_CONNECT is absent from socket module."""
        if hasattr(socket, 'TCP_FASTOPEN_CONNECT'):
            saved = socket.TCP_FASTOPEN_CONNECT
            delattr(socket, 'TCP_FASTOPEN_CONNECT')
        else:
            saved = None
        try:
            conn = self._make_conn()
            conn._apply_socket_options(socket.AF_INET)  # should not raise
        finally:
            if saved is not None:
                socket.TCP_FASTOPEN_CONNECT = saved

    # --- SO_KEEPALIVE ---

    def test_keepalive_enabled(self):
        conn = self._make_conn(tcp_keepalive=True)
        conn._apply_socket_options(socket.AF_INET)
        conn._socket.setsockopt.assert_any_call(
            socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def test_keepalive_disabled(self):
        conn = self._make_conn(tcp_keepalive=False)
        conn._apply_socket_options(socket.AF_INET)
        calls = conn._socket.setsockopt.call_args_list
        for c in calls:
            self.assertNotEqual(c[0][:2], (socket.SOL_SOCKET, socket.SO_KEEPALIVE))

    @patch.object(socket, 'TCP_KEEPIDLE', 4, create=True)
    @patch.object(socket, 'TCP_KEEPINTVL', 5, create=True)
    @patch.object(socket, 'TCP_KEEPCNT', 6, create=True)
    def test_keepalive_tuning(self):
        conn = self._make_conn(tcp_keepalive=True, tcp_keepalive_idle=10, tcp_keepalive_interval=2, tcp_keepalive_count=3)
        conn._apply_socket_options(socket.AF_INET)
        conn._socket.setsockopt.assert_any_call(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
        conn._socket.setsockopt.assert_any_call(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        conn._socket.setsockopt.assert_any_call(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

    # --- TCP_USER_TIMEOUT ---

    @patch.object(socket, 'TCP_USER_TIMEOUT', 18, create=True)
    def test_user_timeout_set(self):
        conn = self._make_conn(tcp_fastopen_connect=False, tcp_keepalive=False, tcp_user_timeout=30000)
        conn._apply_socket_options(socket.AF_INET)
        conn._socket.setsockopt.assert_any_call(
            socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30000)

    @patch.object(socket, 'TCP_USER_TIMEOUT', 18, create=True)
    def test_user_timeout_none_skipped(self):
        conn = self._make_conn(tcp_user_timeout=None)
        conn._apply_socket_options(socket.AF_INET)
        calls = conn._socket.setsockopt.call_args_list
        for c in calls:
            if len(c[0]) >= 2:
                self.assertNotEqual(c[0][1], socket.TCP_USER_TIMEOUT)

    # --- AF_UNIX guard ---

    @unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'AF_UNIX not available')
    def test_af_unix_skips_tcp_options(self):
        conn = self._make_conn(tcp_keepalive=True, tcp_user_timeout=5000)
        conn._apply_socket_options(socket.AF_UNIX)
        # Only sockopts (none here) should have been applied
        conn._socket.setsockopt.assert_not_called()

    # --- User sockopts applied ---

    def test_user_sockopts_applied(self):
        user_opts = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        conn = self._make_conn(sockopts=user_opts)
        conn._apply_socket_options(socket.AF_INET)
        conn._socket.setsockopt.assert_any_call(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # --- OSError suppression ---

    @patch.object(socket, 'TCP_FASTOPEN_CONNECT', 30, create=True)
    def test_oserror_suppressed(self):
        conn = self._make_conn(tcp_fastopen_connect=True, tcp_keepalive=True)
        conn._socket.setsockopt.side_effect = OSError("not supported")
        # Should not raise
        conn._apply_socket_options(socket.AF_INET)


class TestConnectionTcpValidation(unittest.TestCase):
    """Validate Connection.__init__ rejects bad TCP option values."""

    def test_keepalive_idle_bool(self):
        with self.assertRaises(ValueError):
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_keepalive_idle=True)

    def test_keepalive_interval_bool(self):
        with self.assertRaises(ValueError):
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_keepalive_interval=False)

    def test_keepalive_count_negative(self):
        with self.assertRaises(ValueError):
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_keepalive_count=-1)

    def test_user_timeout_bool(self):
        with self.assertRaises(ValueError):
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_user_timeout=True)

    def test_user_timeout_float(self):
        with self.assertRaises(ValueError):
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_user_timeout=1.5)

    def test_user_timeout_zero_allowed(self):
        # Should not raise - validation runs before connect attempt
        try:
            Connection.__new__(Connection).__init__('127.0.0.1', tcp_user_timeout=0)
        except ValueError:
            self.fail("tcp_user_timeout=0 should be allowed")
        except Exception:
            pass  # Other errors (connection, etc.) are expected


class TestClusterTcpOptions(unittest.TestCase):
    """Unit tests for Cluster-level TCP option passthrough."""

    def test_defaults_propagated(self):
        c = Cluster(contact_points=[])
        self.assertFalse(c.tcp_fastopen_connect)
        self.assertFalse(c.tcp_keepalive)
        self.assertEqual(c.tcp_keepalive_idle, 30)
        self.assertEqual(c.tcp_keepalive_interval, 5)
        self.assertEqual(c.tcp_keepalive_count, 6)
        self.assertIsNone(c.tcp_user_timeout)

    def test_custom_values(self):
        c = Cluster(contact_points=[], tcp_keepalive_idle=60, tcp_user_timeout=15000)
        self.assertEqual(c.tcp_keepalive_idle, 60)
        self.assertEqual(c.tcp_user_timeout, 15000)

    def test_make_connection_kwargs_passthrough(self):
        c = Cluster(contact_points=[], tcp_keepalive_idle=45)
        ep = DefaultEndPoint('127.0.0.1', 9042)
        kw = c._make_connection_kwargs(ep, {})
        self.assertEqual(kw['tcp_keepalive_idle'], 45)
        self.assertFalse(kw['tcp_fastopen_connect'])

    def test_tcp_user_timeout_derived_from_heartbeat(self):
        c = Cluster(contact_points=[], idle_heartbeat_timeout=30)
        ep = DefaultEndPoint('127.0.0.1', 9042)
        kw = c._make_connection_kwargs(ep, {})
        self.assertEqual(kw['tcp_user_timeout'], 30000)

    def test_tcp_user_timeout_explicit_overrides_derivation(self):
        c = Cluster(contact_points=[], idle_heartbeat_timeout=30, tcp_user_timeout=5000)
        ep = DefaultEndPoint('127.0.0.1', 9042)
        kw = c._make_connection_kwargs(ep, {})
        self.assertEqual(kw['tcp_user_timeout'], 5000)

    def test_tcp_user_timeout_not_derived_when_heartbeat_zero(self):
        c = Cluster(contact_points=[], idle_heartbeat_timeout=0)
        ep = DefaultEndPoint('127.0.0.1', 9042)
        kw = c._make_connection_kwargs(ep, {})
        self.assertIsNone(kw['tcp_user_timeout'])

    def test_tcp_user_timeout_not_derived_when_heartbeat_interval_zero(self):
        c = Cluster(contact_points=[], idle_heartbeat_interval=0, idle_heartbeat_timeout=30)
        ep = DefaultEndPoint('127.0.0.1', 9042)
        kw = c._make_connection_kwargs(ep, {})
        self.assertIsNone(kw['tcp_user_timeout'])

    def test_validation_keepalive_idle_negative(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_idle=-1)

    def test_validation_keepalive_idle_zero(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_idle=0)

    def test_validation_keepalive_idle_float(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_idle=1.5)

    def test_validation_keepalive_interval_negative(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_interval=-1)

    def test_validation_keepalive_count_zero(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_count=0)

    def test_validation_user_timeout_negative(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_user_timeout=-1)

    def test_validation_user_timeout_zero_allowed(self):
        c = Cluster(contact_points=[], tcp_user_timeout=0)
        self.assertEqual(c.tcp_user_timeout, 0)

    def test_validation_user_timeout_float(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_user_timeout=30.5)

    def test_validation_keepalive_idle_bool(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_idle=True)

    def test_validation_keepalive_interval_bool(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_interval=True)

    def test_validation_keepalive_count_bool(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_keepalive_count=False)

    def test_validation_user_timeout_bool(self):
        with self.assertRaises(ValueError):
            Cluster(contact_points=[], tcp_user_timeout=True)


class TestSocketOptionsPrecedence(unittest.TestCase):
    """Verify user sockopts override built-in TCP options."""

    def _make_conn(self, **overrides):
        conn = object.__new__(Connection)
        conn.sockopts = None
        conn.tcp_fastopen_connect = False
        conn.tcp_keepalive = False
        conn.tcp_keepalive_idle = 30
        conn.tcp_keepalive_interval = 5
        conn.tcp_keepalive_count = 6
        conn.tcp_user_timeout = None
        for k, v in overrides.items():
            setattr(conn, k, v)
        conn._socket = Mock()
        return conn

    def test_user_sockopts_override_builtin_keepalive(self):
        """User SO_KEEPALIVE=0 should override built-in SO_KEEPALIVE=1."""
        user_opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 0)]
        conn = self._make_conn(tcp_keepalive=True, sockopts=user_opts)
        conn._apply_socket_options(socket.AF_INET)
        # Last call with SO_KEEPALIVE should be the user's (0), not built-in (1)
        keepalive_calls = [
            c for c in conn._socket.setsockopt.call_args_list
            if len(c[0]) >= 2 and c[0][0] == socket.SOL_SOCKET and c[0][1] == socket.SO_KEEPALIVE
        ]
        self.assertTrue(len(keepalive_calls) >= 2, "Expected both built-in and user SO_KEEPALIVE calls")
        self.assertEqual(keepalive_calls[-1][0][2], 0, "User sockopt should be last and win")

    @unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'AF_UNIX not available')
    def test_user_sockopts_still_applied_for_af_unix(self):
        """Even AF_UNIX should get user sockopts applied."""
        user_opts = [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
        conn = self._make_conn(sockopts=user_opts)
        conn._apply_socket_options(socket.AF_UNIX)
        conn._socket.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

