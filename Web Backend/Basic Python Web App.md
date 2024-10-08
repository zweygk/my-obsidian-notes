
Flask boilerplate code for serving json

```python
from flask import Flask, request, Response, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

@app.route('/', methods=['GET'])
def health():
    # Response with 200 to GET request (health check). Separated so as to not require auth.
    return Response('{"ok":"ok"}', status=200, mimetype='application/json')


@app.route('/', methods=['POST'])
def main():
	# do stuff
    return json_output


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
```

