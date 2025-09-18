"""
Test for __slots__ implementation in connection module classes.
This test ensures that memory optimization via __slots__ is working correctly.
"""
import unittest
from cassandra.connection import (
    EndPoint, DefaultEndPoint, SniEndPoint, UnixSocketEndPoint,
    _Frame, ContinuousPagingSession, ShardawarePortGenerator,
    _ConnectionIOBuffer, ResponseWaiter, HeartbeatFuture, Timer, TimerManager
)


class SlotsImplementationTest(unittest.TestCase):
    """Test that targeted classes have __slots__ and prevent dynamic attributes."""

    def test_endpoint_classes_have_slots(self):
        """Test EndPoint and its subclasses have __slots__ implemented."""
        # EndPoint base class should have empty slots
        self.assertEqual(EndPoint.__slots__, ())
        
        # Test DefaultEndPoint
        ep = DefaultEndPoint('127.0.0.1', 9042)
        self.assertFalse(hasattr(ep, '__dict__'))
        with self.assertRaises(AttributeError):
            ep.dynamic_attr = 'test'
            
        # Test SniEndPoint
        sni_ep = SniEndPoint('proxy.example.com', 'server.example.com', 9042)
        self.assertFalse(hasattr(sni_ep, '__dict__'))
        with self.assertRaises(AttributeError):
            sni_ep.dynamic_attr = 'test'
            
        # Test UnixSocketEndPoint
        unix_ep = UnixSocketEndPoint('/tmp/cassandra.sock')
        self.assertFalse(hasattr(unix_ep, '__dict__'))
        with self.assertRaises(AttributeError):
            unix_ep.dynamic_attr = 'test'

    def test_frame_class_has_slots(self):
        """Test _Frame class has __slots__ implemented."""
        frame = _Frame(4, 0, 1, 7, 9, 100)
        self.assertFalse(hasattr(frame, '__dict__'))
        with self.assertRaises(AttributeError):
            frame.dynamic_attr = 'test'
            
        # Test that all expected attributes are accessible
        self.assertEqual(frame.version, 4)
        self.assertEqual(frame.flags, 0)
        self.assertEqual(frame.stream, 1)
        self.assertEqual(frame.opcode, 7)
        self.assertEqual(frame.body_offset, 9)
        self.assertEqual(frame.end_pos, 100)

    def test_timer_classes_have_slots(self):
        """Test Timer and TimerManager classes have __slots__ implemented."""
        # Test Timer
        timer = Timer(5.0, lambda: None)
        self.assertFalse(hasattr(timer, '__dict__'))
        with self.assertRaises(AttributeError):
            timer.dynamic_attr = 'test'
            
        # Test Timer attributes
        self.assertEqual(timer.canceled, False)
        self.assertIsNotNone(timer.end)
        self.assertIsNotNone(timer.callback)
        
        # Test TimerManager
        timer_mgr = TimerManager()
        self.assertFalse(hasattr(timer_mgr, '__dict__'))
        with self.assertRaises(AttributeError):
            timer_mgr.dynamic_attr = 'test'

    def test_utility_classes_have_slots(self):
        """Test utility classes have __slots__ implemented."""
        # Test ShardawarePortGenerator
        self.assertEqual(ShardawarePortGenerator.__slots__, ())
        
        # Test _ConnectionIOBuffer
        class MockConnection:
            pass
            
        io_buffer = _ConnectionIOBuffer(MockConnection())
        self.assertFalse(hasattr(io_buffer, '__dict__'))
        with self.assertRaises(AttributeError):
            io_buffer.dynamic_attr = 'test'
            
        # Test ResponseWaiter
        response_waiter = ResponseWaiter(MockConnection(), 2, True)
        self.assertFalse(hasattr(response_waiter, '__dict__'))
        with self.assertRaises(AttributeError):
            response_waiter.dynamic_attr = 'test'

    def test_slots_prevent_memory_overhead(self):
        """Test that objects with __slots__ don't have __dict__ overhead."""
        instances = [
            DefaultEndPoint('127.0.0.1', 9042),
            SniEndPoint('proxy.example.com', 'server.example.com', 9042),
            UnixSocketEndPoint('/tmp/cassandra.sock'),
            _Frame(4, 0, 1, 7, 9, 100),
            Timer(5.0, lambda: None),
            TimerManager(),
        ]
        
        for instance in instances:
            with self.subTest(instance=instance.__class__.__name__):
                # Ensure no __dict__ is present (memory optimization)
                self.assertFalse(hasattr(instance, '__dict__'))
                # Ensure __slots__ is defined
                self.assertTrue(hasattr(instance.__class__, '__slots__'))


if __name__ == '__main__':
    unittest.main()