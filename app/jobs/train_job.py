import asyncio
import logging
from app.server.database import bank_collection
from app.ai.train import handle_label

logger = logging.getLogger(__name__)


async def _safe_handle_label(bank_id: str):
    try:
        await handle_label(bank_id)
        logger.info(f"✅ Train success: bank={bank_id}")
    except Exception as e:
        logger.exception(f"❌ Train failed: bank={bank_id} | {e}")


async def train_all_banks():
    banks = await bank_collection.find().to_list(None)

    if not banks:
        logger.warning("⚠️ No banks found")
        return

    queued = 0

    for bank in banks:
        bank_id = str(bank["_id"])

        asyncio.create_task(
            _safe_handle_label(bank_id)
        )

        queued += 1

    logger.info(f"🚀 Queued {queued} training jobs")
