from motor.motor_asyncio import AsyncIOMotorClient

MONGO_DETAILS = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_DETAILS)

database = client.edly

bank_collection = database.get_collection("banks")
label_collection = database.get_collection("labels")
question_collection = database.get_collection("questions")