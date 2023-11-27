from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from ai_model import AIModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


@api.route('/predict')
class Predict(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        data = request.files['image'].read()
        temp = AIModel().predict(data)
        temp = jsonify(temp)
        return temp

