from motor.motor_asyncio import AsyncIOMotorClient

MONGO_DETAILS = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_DETAILS)

database = client.edly

bank_collection = database.get_collection("banks")
bank_model_collection = database.get_collection("bank_models")
label_collection = database.get_collection("labels")
question_collection = database.get_collection("questions")

train_run_collection = database.get_collection("train_runs")
question_train_state_collection = database.get_collection("question_trains") 
