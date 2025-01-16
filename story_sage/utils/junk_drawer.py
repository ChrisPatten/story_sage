"""Utility functions that don't fit anywhere else"""

import logging

def format_log_with_emoji(level_name: str) -> str:
    """Returns appropriate emoji for different log levels"""
    emoji_map = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    return emoji_map.get(level_name, '📝')

class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji to log messages"""
    def format(self, record):
        # Add emoji based on log level
        emoji = format_log_with_emoji(record.levelname)
        record.emoji = emoji * 6
        return super().format(record)