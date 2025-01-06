from flask import Flask
from routes import app

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the Flask app with custom configurations.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP address.')
    parser.add_argument('--port', type=int, default=5000, help='Port number.')
    parser.add_argument('--file', type=str, default='qa_database.jsonl', help='Path to the JSONL file.')
    args = parser.parse_args()

    # Pass custom file path to the app
    app.config['JSONL_FILE_PATH'] = args.file
    app.run(host=args.host, port=args.port, debug=True)
