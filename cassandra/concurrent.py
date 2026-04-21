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


from collections import deque, namedtuple
from heapq import heappush, heappop
from itertools import cycle
from threading import Condition, Event, Thread
import sys

from cassandra.cluster import ResultSet, EXEC_PROFILE_DEFAULT

import logging
log = logging.getLogger(__name__)


ExecutionResult = namedtuple('ExecutionResult', ['success', 'result_or_exc'])

def execute_concurrent(session, statements_and_parameters, concurrency=100, raise_on_first_error=True, results_generator=False, execution_profile=EXEC_PROFILE_DEFAULT):
    """
    Executes a sequence of (statement, parameters) tuples concurrently.  Each
    ``parameters`` item must be a sequence or :const:`None`.

    The `concurrency` parameter controls how many statements will be executed
    concurrently.

    If `raise_on_first_error` is left as :const:`True`, execution will stop
    after the first failed statement and the corresponding exception will be
    raised.

    `results_generator` controls how the results are returned.

    * If :const:`False`, the results are returned only after all requests have completed.
    * If :const:`True`, a generator expression is returned. Using a generator results in a constrained
      memory footprint when the results set will be large -- results are yielded
      as they return instead of materializing the entire list at once. The trade for lower memory
      footprint is marginal CPU overhead (more thread coordination and sorting out-of-order results
      on-the-fly).

    `execution_profile` argument is the execution profile to use for this
    request, it is passed directly to :meth:`Session.execute_async`.

    A sequence of ``ExecutionResult(success, result_or_exc)`` namedtuples is returned
    in the same order that the statements were passed in.  If ``success`` is :const:`False`,
    there was an error executing the statement, and ``result_or_exc`` will be
    an :class:`Exception`.  If ``success`` is :const:`True`, ``result_or_exc``
    will be the query result.

    Example usage::

        select_statement = session.prepare("SELECT * FROM users WHERE id=?")

        statements_and_params = []
        for user_id in user_ids:
            params = (user_id, )
            statements_and_params.append((select_statement, params))

        results = execute_concurrent(
            session, statements_and_params, raise_on_first_error=False)

        for (success, result) in results:
            if not success:
                handle_error(result)  # result will be an Exception
            else:
                process_user(result[0])  # result will be a list of rows

    Note: in the case that `generators` are used, it is important to ensure the consumers do not
    block or attempt further synchronous requests, because no further IO will be processed until
    the consumer returns. This may also produce a deadlock in the IO event thread.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")

    if not statements_and_parameters:
        return []

    executor = ConcurrentExecutorGenResults(session, statements_and_parameters, execution_profile) \
        if results_generator else ConcurrentExecutorListResults(session, statements_and_parameters, execution_profile)
    return executor.execute(concurrency, raise_on_first_error)


class _ConcurrentExecutor(object):

    def __init__(self, session, statements_and_params, execution_profile):
        self.session = session
        self._enum_statements = enumerate(iter(statements_and_params))
        self._execution_profile = execution_profile
        self._condition = Condition()
        self._fail_fast = False
        self._results_queue = []
        self._current = 0
        self._exec_count = 0
        self._executing = False

    def execute(self, concurrency, fail_fast):
        self._fail_fast = fail_fast
        self._results_queue = []
        self._current = 0
        self._exec_count = 0
        with self._condition:
            for n in range(concurrency):
                if not self._execute_next():
                    break
        return self._results()

    def _execute_next(self):
        # lock must be held
        try:
            (idx, (statement, params)) = next(self._enum_statements)
            self._exec_count += 1
            self._execute(idx, statement, params)
            return True
        except StopIteration:
            pass

    def _execute(self, idx, statement, params):
        # When execute_async completes synchronously (e.g. immediate timeout),
        # the errback fires inline: _on_error -> _put_result -> _execute_next
        # -> _execute.  Without protection this recurses once per remaining
        # statement and blows the stack.
        #
        # ``_executing`` marks that we are already inside this method higher up
        # the call stack.  When a synchronous callback re-enters, we just stash
        # the pending work in ``_pending_executions`` and let the outermost
        # invocation drain it in a loop -- no recursion.
        if self._executing:
            self._pending_executions.append((idx, statement, params))
            return

        self._executing = True
        self._pending_executions = [(idx, statement, params)]
        try:
            while self._pending_executions:
                p_idx, p_statement, p_params = self._pending_executions.pop(0)
                try:
                    future = self.session.execute_async(p_statement, p_params, timeout=None, execution_profile=self._execution_profile)
                    args = (future, p_idx)
                    future.add_callbacks(
                        callback=self._on_success, callback_args=args,
                        errback=self._on_error, errback_args=args)
                except Exception as exc:
                    self._put_result(exc, p_idx, False)
        finally:
            self._executing = False

    def _on_success(self, result, future, idx):
        future.clear_callbacks()
        self._put_result(ResultSet(future, result), idx, True)

    def _on_error(self, result, future, idx):
        self._put_result(result, idx, False)


class ConcurrentExecutorGenResults(_ConcurrentExecutor):

    def _put_result(self, result, idx, success):
        with self._condition:
            heappush(self._results_queue, (idx, ExecutionResult(success, result)))
            self._execute_next()
            self._condition.notify()

    def _results(self):
        with self._condition:
            while self._current < self._exec_count:
                while not self._results_queue or self._results_queue[0][0] != self._current:
                    self._condition.wait()
                while self._results_queue and self._results_queue[0][0] == self._current:
                    _, res = heappop(self._results_queue)
                    try:
                        self._condition.release()
                        if self._fail_fast and not res[0]:
                            raise res[1]
                        yield res
                    finally:
                        self._condition.acquire()
                    self._current += 1


class ConcurrentExecutorListResults(_ConcurrentExecutor):

    _exception = None

    def execute(self, concurrency, fail_fast):
        self._exception = None
        self._submit_ready = deque()
        self._submit_event = Event()
        self._submit_stopped = False
        # Submit the initial batch from the calling thread (no contention
        # yet -- the submitter thread is not started until afterward).
        result = super(ConcurrentExecutorListResults, self).execute(concurrency, fail_fast)
        return result

    def _results(self):
        # Start the submitter thread *after* the initial batch has been
        # fully dispatched so that _enum_statements and _exec_count are
        # not accessed concurrently during the seeding phase.
        self._submitter = Thread(target=self._submitter_loop,
                                 daemon=True, name="concurrent-submitter")
        self._submitter.start()

        with self._condition:
            while self._current < self._exec_count:
                self._condition.wait()
                if self._exception and self._fail_fast:
                    break
        self._submit_stopped = True
        self._submit_event.set()
        if self._exception and self._fail_fast:
            raise self._exception
        return [r[1] for r in sorted(self._results_queue)]

    def _put_result(self, result, idx, success):
        self._results_queue.append((idx, ExecutionResult(success, result)))
        if not success and self._fail_fast:
            if not self._exception:
                self._exception = result
        # Signal the submitter thread to send the next request instead of
        # calling _execute_next() inline.  This keeps the event-loop thread
        # (which fires the callback) free to process I/O rather than doing
        # query-plan lookup, message serialisation, and connection borrowing.
        self._submit_ready.append(1)
        self._submit_event.set()

    def _submitter_loop(self):
        """Drain completion signals and submit follow-up requests.

        Runs on a dedicated thread so that the libev event-loop thread
        only needs to do the lightweight ``deque.append`` + ``Event.set``
        in ``_put_result`` rather than the full execute_async cycle
        (query-plan, borrow connection, serialise, enqueue).

        Calls execute_async directly instead of going through the
        _execute / _execute_next indirection to avoid per-request
        overhead from the re-entrancy guard and pending-executions list.
        """
        ready = self._submit_ready
        ready_event = self._submit_event
        enum_stmts = self._enum_statements
        session = self.session
        profile = self._execution_profile
        on_success = self._on_success
        on_error = self._on_error
        exec_count = self._exec_count  # snapshot after initial batch
        exhausted = False
        while not self._submit_stopped:
            ready_event.wait()
            ready_event.clear()
            count = 0
            while True:
                try:
                    ready.popleft()
                    count += 1
                except IndexError:
                    break
            if count == 0:
                continue
            # Submit follow-up requests directly (fast path).
            # The iterator is only consumed from this thread (the initial
            # batch was fully dispatched before this thread started).
            if not exhausted:
                for _ in range(count):
                    try:
                        idx, (statement, params) = next(enum_stmts)
                    except StopIteration:
                        exhausted = True
                        break
                    exec_count += 1
                    try:
                        future = session.execute_async(statement, params,
                                                       timeout=None,
                                                       execution_profile=profile)
                        args = (future, idx)
                        future.add_callbacks(
                            callback=on_success, callback_args=args,
                            errback=on_error, errback_args=args)
                    except Exception as exc:
                        self._put_result(exc, idx, False)
            with self._condition:
                self._exec_count = exec_count
                self._current += count
                if self._current >= self._exec_count:
                    self._condition.notify()
                if self._exception and self._fail_fast:
                    self._condition.notify()



def execute_concurrent_with_args(session, statement, parameters, *args, **kwargs):
    """
    Like :meth:`~cassandra.concurrent.execute_concurrent()`, but takes a single
    statement and a sequence of parameters.  Each item in ``parameters``
    should be a sequence or :const:`None`.

    Example usage::

        statement = session.prepare("INSERT INTO mytable (a, b) VALUES (1, ?)")
        parameters = [(x,) for x in range(1000)]
        execute_concurrent_with_args(session, statement, parameters, concurrency=50)
    """
    return execute_concurrent(session, zip(cycle((statement,)), parameters), *args, **kwargs)
