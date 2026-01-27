# core/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging

from app.jobs.train_job import train_all_banks

logger = logging.getLogger(__name__)
 
scheduler = AsyncIOScheduler(
    timezone="Asia/Ho_Chi_Minh"
)

def register_jobs():
    scheduler.add_job(
        train_all_banks,
        CronTrigger(hour=2, minute=0),
        id="nightly_train",
        replace_existing=True,
        max_instances=1,      
        misfire_grace_time=300   
    )
    logger.info("✅ Công việc lập lịch đã được đăng ký")

def start_scheduler():
    if not scheduler.running:
        register_jobs()
        scheduler.start()
        logger.info("🚀 Trình lập lịch đã bắt đầu")

def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 Trình lập lịch đã dừng")
