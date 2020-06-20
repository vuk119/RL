import unittest

from easy21 import Easy21

class TestEasy21(unittest.TestCase):

    def setUp(self):
        self.env = Easy21()

    def test_reset(self):
        (player_state, dealer_state) = self.env.reset()

        self.assertGreater(player_state, 0)
        self.assertGreater(-player_state, -11)

        self.assertGreater(dealer_state, 0)
        self.assertGreater(-dealer_state, -11)

    def test_draw_card(self):
        card_value = self.env.draw_card()

        self.assertGreater(card_value, 0)
        self.assertGreater(-card_value, -11)

if __name__ == '__main__':
    unittest.main()
