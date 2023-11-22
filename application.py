from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from ai_model import AIModel

app = Flask(__name__)
api = Api(app)


@api.route('/predict')
class Predict(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        data = request.json.get('data')
        temp = AIModel().predict(data)
        temp = jsonify(temp)
        return temp

