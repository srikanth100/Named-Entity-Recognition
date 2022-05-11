import pickle
from flask import Flask, jsonify
from flask_ngrok import run_with_ngrok
from flask import request
import json
from flask_cors import CORS, cross_origin
import warnings
warnings.filterwarnings('ignore')
from project import MODELS_DIR

model_file = open(MODELS_DIR + '\crf', 'rb')
model = pickle.load(model_file)
model_file.close()
