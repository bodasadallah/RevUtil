import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-9b-it",
        help="Model name",)
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
        "--prompt",
        type=str,
        default="ternary_score_prompt",
        help="Prompt",)
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path",)
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path",)
    parser.add_argument(
        "--review_field",
        type=str,
        default="review_point",
        help="Review field",)
    parser.add_argument(
        "--total_points",
        type=int,
        default=0,
        help="Total points to evaluate, if zero, then it will be set to the len of the dataframe",)
    parser.add_argument(
        '--aspect',
        type=str,
        default='all',
        help='Aspect to evaluate',)
    
    parser.add_argument(
        '--run_statistics',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--compare_to_human',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--statistics_path',
        type=str,
        default=None
    )

    

    
    
    
