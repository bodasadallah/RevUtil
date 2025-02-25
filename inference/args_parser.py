import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="allenai/tulu-2-dpo-70b",
        help="Tokenizer name",)
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="allenai/scitulu-7b",
        help="Model name",)
    parser.add_argument(
        "--finetune_model_name",
        type=str,
        default="/l/users/abdelrahman.sadallah/review_evaluation/actionability/checkpoint-33/",
        help="Model name",)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="boda/review_evaluation_fine_tuning",
        help="Dataset name",)
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="actionability",
        help="Aspect",)
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max number of tokens to generate",)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling",)
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top p for sampling",)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",)
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="Output path",)
    parser.add_argument(
        '--aspect',
        type=str,
        default='all',
        help='Aspect to evaluate',)
    

    
    
    
