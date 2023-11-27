from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource
from ai_model import AIModel
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)
api = Api(app)


@api.route('/predict')
class Predict(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        img = BytesIO(request.files['image'].read())
        #temp = AIModel().predict(img)
        temp = {'age':'obs', 'gender':'women', 'emotion':'angry'}
        temp = jsonify(temp)
        # Create a response with the entire JSON string and close the connection
        response = make_response(temp)
        response.headers['Content-Length'] = len(response.data)
        return response

