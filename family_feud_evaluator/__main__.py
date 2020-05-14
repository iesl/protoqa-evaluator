import argparse

from .data_processing import *


def main():
    parsers = dict()
    parser = argparse.ArgumentParser()
    command_subparser = parser.add_subparsers(help="Command", dest="command")

    convert_parser = command_subparser.add_parser("convert", help="Convert files")
    convert_parser_sp = convert_parser.add_subparsers(
        help="Type of conversion", dest="type"
    )

    clustered_data_parser = convert_parser_sp.add_parser(
        "clustered", help="Convert clustering XLSX files to JSONL"
    )
    clustered_data_parser.add_argument("xlsx_files", type=Path, nargs="+")
    clustered_data_parser.add_argument("--output_jsonl", type=Path, required=True)

    ranking_data_parser = convert_parser_sp.add_parser(
        "ranking", help="Convert ranking XLSX files to JSONL"
    )
    ranking_data_parser.add_argument("xlsx_file", type=Path)
    ranking_data_parser.add_argument("--question_jsonl", type=Path, required=True)
    ranking_data_parser.add_argument("--output_jsonl", type=Path, required=True)
    ranking_data_parser.add_argument("--include_do_not_use", action="store_true")
    ranking_data_parser.add_argument("--allow_incomplete", action="store_true")

    ranking_comparison_data_parser = convert_parser_sp.add_parser(
        "ranking_comparison", help="Convert ranking comparison XLSX files to JSONL"
    )
    ranking_comparison_data_parser.add_argument("xlsx_file", type=Path)
    # ranking_data_parser.add_argument('--question_jsonl', type=Path, required=True)
    ranking_comparison_data_parser.add_argument(
        "--output_jsonl", type=Path, required=True
    )
    # ranking_data_parser.add_argument('--include_do_not_use')

    args = parser.parse_args()

    if args.command == "convert":
        if args.type == "clustered":
            convert_clustered(vars(args))
        if args.type == "ranking":
            convert_ranking(vars(args))
        if args.type == "ranking_comparison":
            convert_ranking_comparison(vars(args))


def convert_clustered(config):
    # fix up the file paths, in case they contain ~
    for i, file in enumerate(config["xlsx_files"]):
        file = file.expanduser()
        assert file.exists()
        config["xlsx_files"][i] = file
    config["output_jsonl"] = config["output_jsonl"].expanduser()

    q = dict()
    for idx, xlsx_file in enumerate(config["xlsx_files"]):
        next_q = load_data_from_excel(xlsx_file, idx + 1)
        q.update(next_q)
    return save_to_jsonl(config["output_jsonl"], q)


def convert_ranking(config):
    for k in ["xlsx_file", "question_jsonl", "output_jsonl"]:
        config[k] = config[k].expanduser()

    ranking_data = load_ranking_data(config["xlsx_file"])
    question_data = load_data_from_jsonl(config["question_jsonl"])
    answers_dict = convert_ranking_data_to_answers(
        ranking_data, question_data, config["allow_incomplete"]
    )
    if not config["include_do_not_use"]:
        do_not_use = {k for k, v in question_data.items() if v["do-not-use"]}
        num_int = do_not_use.intersection(answers_dict.keys())
        # logging.info(f'Removing {num_int} answers whose associated questions were marked as DO_NOT_USE')
        print(
            f"Removing {num_int} answers whose associated questions were marked as DO_NOT_USE"
        )
        answers_dict = {k: v for k, v in answers_dict.items() if k not in do_not_use}
    return save_to_jsonl(config["output_jsonl"], answers_dict)


def convert_ranking_comparison(config):
    for k in ["xlsx_file", "output_jsonl"]:
        config[k] = config[k].expanduser()

    ranking_data = load_ranking_comparison_data(config["xlsx_file"])
    return save_to_jsonl(config["output_jsonl"], ranking_data)


if __name__ == "__main__":
    main()
