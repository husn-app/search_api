import unittest
import json
from app import app

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_api_query(self):
        response = self.app.post('/api/query', json={'query': 'red dress'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(set(data.keys()), {'query', 'products', 'scores'})

    def test_api_query_empty(self):
        response = self.app.post('/api/query', json={'query': ''})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()