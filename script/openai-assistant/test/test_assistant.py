# test_assistant.py
import unittest
from unittest.mock import patch, MagicMock
from src.assistant import (
    update_assistant_list,
    manage_assistant_file,
    list_assistant_file_names,
    get_assitant_files,  
    create_and_dump_messages,
    find_assistant_by_name
)

class TestAssistant(unittest.TestCase):

    def setUp(self):
        # Example assistant object for testing
        self.example_assistant = {
            "id": "test_id",
            "name": "Test Assistant",
            "description": "A test assistant.",
            "model": "gpt-4",
            "instructions": "Test instructions.",
            "tools": [],
            "file_ids": [],
            "metadata": {}
        }

        self.file_path = './dummy_assistant_list.json'
        self.assistant_file_path = './dummy_assistant_files/'
    
    # def tearDown(self):
    #     # Reset the dummy JSON file to an empty structure after each test
    #     with open(self.file_path, 'w') as file:
    #         file.write('{"data": []}')

    def test_update_assistant_list(self):
        list_file = {"data": []}
        updated_list = update_assistant_list(list_file, self.example_assistant)
        self.assertIn(self.example_assistant, updated_list['data'])

    @patch('src.assistant.write_json_file')
    @patch('src.assistant.read_json_file')
    def test_manage_assistant_file(self, mock_read_json, mock_write_json):
        mock_read_json.return_value = ({'data': []}, None)
        mock_write_json.return_value = (True, None)
        updated_list, error = manage_assistant_file(self.file_path, self.example_assistant)
        self.assertIsNone(error)
        self.assertIn(self.example_assistant, updated_list['data'])

    @patch('os.listdir')
    @patch('os.path.isfile')
    def test_list_assistant_file_names(self, mock_isfile, mock_listdir):
        mock_listdir.return_value = ['dummy1.pdf', 'dummy2.pdf']
        mock_isfile.side_effect = lambda x: x != 'dir1'
        file_names = list_assistant_file_names(self.assistant_file_path)
        self.assertEqual(file_names, ['dummy1.pdf', 'dummy2.pdf'])

    def test_get_assitant_files(self):
        json_data = {
            "data": [
                {"id": "file_1", "purpose": "assistants"},
                {"id": "file_2", "purpose": "other"}
            ]
        }
        retrieved_files = get_assitant_files(json_data)
        self.assertIn("file_1", retrieved_files)
        self.assertNotIn("file_2", retrieved_files)

    @patch('os.path.isfile')
    @patch('src.assistant.write_json_file')
    @patch('json.load')
    def test_create_and_dump_messages(self, mock_json_load, mock_write_json, mock_isfile):
        mock_isfile.return_value = True
        mock_json_load.return_value = []
        mock_write_json.side_effect = lambda file_path, data: (True, None)

        messages = [{"msg": "Hello, world!"}]
        create_and_dump_messages(messages, self.file_path)

        # Since the function doesn't return anything, we check the call to write_json_file
        mock_write_json.assert_called_once()
        args, kwargs = mock_write_json.call_args
        file_path_arg, data_arg = args

        # Check if the file path argument is correct
        self.assertEqual(file_path_arg, self.file_path)

        # Check if the data written to the file is as expected
        self.assertEqual(data_arg, messages)

    @patch('src.assistant.read_json_file')
    def test_find_assistant_by_name(self, mock_read_json):
        mock_read_json.return_value = ({'data': [self.example_assistant]}, None)
        assistant_id = find_assistant_by_name(self.file_path, "Test Assistant")
        self.assertEqual(assistant_id, "test_id")

        # Test for a non-existent assistant name
        assistant_id = find_assistant_by_name(self.file_path, "Non-Existent Assistant")
        self.assertIsNone(assistant_id)

if __name__ == '__main__':
    unittest.main()
