import torch
import logging
import sys

from .utils import setup_logger, get_optimal_device
from .cli import create_arg_parser, demo

def main():
    """Parse arguments and run the demo"""
    # Set up logger
    logger = setup_logger()
    
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Validate device setting
    try:
        args.device = get_optimal_device(args.device)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Run the demo
    demo(args)


if __name__ == "__main__":
    main()