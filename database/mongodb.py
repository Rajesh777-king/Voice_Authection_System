from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["speaker_auth"]

users_collection = db["users"]