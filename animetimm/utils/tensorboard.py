import logging
import os

from tensorboard.backend.event_processing import event_accumulator


def is_tensorboard_has_content(log_file_path):
    if not os.path.exists(log_file_path):
        return False
    if os.path.getsize(log_file_path) == 0:
        return False

    try:
        ea = event_accumulator.EventAccumulator(log_file_path)
        ea.Reload()

        has_content = (
                len(ea.Tags().get('scalars', [])) > 0 or
                len(ea.Tags().get('images', [])) > 0 or
                len(ea.Tags().get('histograms', [])) > 0 or
                len(ea.Tags().get('audio', [])) > 0 or
                len(ea.Tags().get('tensors', [])) > 0 or
                ea.Tags().get('graph', False)
        )

        return has_content

    except Exception as e:
        logging.exception(f'Error occurred when analysising tensorboard file {log_file_path}.')
        return False
