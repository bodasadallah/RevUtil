import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--finetuning_type",
        type=str,
        help="Full or Adapters",)
    
    parser.add_argument(
        "--step",
        type=str,
        help="What step to take the checkpoint for",)
    parser.add_argument(
        "--checkpoint_parent_path",
        type=str,
        help="Parent path for the checkpoints",)
    

    parser.add_argument(
        "--generation_type",
        type=str,
        default="score_only",
        help="Whether to only output scores, or also output rationales.",)
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="chat",
        help="Whether to use Instruction Prompting or Chat Prompting")


    parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default="1",
    help="Tensor parallel size",)
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="allenai/tulu-2-dpo-70b",
        help="Tokenizer name",)
    parser.add_argument(
        "--full_model_name",
        type=str,
        default="allenai/scitulu-7b",
        help="Model name",)
    parser.add_argument(
        "--finetune_model_name",
        type=str,
        help="Model name",
        default=None,)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="boda/review_evaluation_fine_tuning",
        help="Dataset name",)
    
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="actionability",
        help="Aspect",)
    
    parser.add_argument(
        "--gold_label_format",
        type=str,
        default='chatgpt_ASPECT_score',
        help="format for the gold label",)

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
    

    
    
    
