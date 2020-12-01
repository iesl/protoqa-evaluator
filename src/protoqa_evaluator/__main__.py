import click


@click.group()
def main():
    """Evaluation and data processing for ProtoQA common sense QA dataset"""
    pass


@main.group()
def convert():
    pass


@convert.command()
@click.argument("xlsx_files", type=click.Path(exists=True), nargs=-1)
@click.argument("--output_jsonl", type=click.Path(), nargs=1)
def clustered(xlsx_files, output_jsonl):
    "Convert clustering XLSX files to JSONL"
    from .data_processing import load_data_from_excel, save_to_jsonl

    q = dict()
    for idx, xlsx_file in enumerate(xlsx_files):
        next_q = load_data_from_excel(xlsx_file, idx + 1)
        q.update(next_q)
    return save_to_jsonl(output_jsonl, q)


@convert.command()
@click.argument("xlsx_file", type=click.Path())
@click.argument("question_jsonl", type=click.Path())
@click.argument("output_jsonl", type=click.Path())
@click.option("--include_do_not_use/--exclude_do_not_use", default=False)
@click.option("--allow_incomplete/--no_incomplete", default=False)
def ranking(
    xlsx_file, question_jsonl, output_jsonl, include_do_not_use, allow_incomplete
):
    """Convert ranking XLSX files to JSONL"""
    from .data_processing import (
        load_ranking_data,
        load_data_from_jsonl,
        convert_ranking_data_to_answers,
        save_to_jsonl,
    )

    ranking_data = load_ranking_data(xlsx_file)
    question_data = load_data_from_jsonl(question_jsonl)
    answers_dict = convert_ranking_data_to_answers(
        ranking_data, question_data, allow_incomplete
    )
    if not include_do_not_use:
        do_not_use = {k for k, v in question_data.items() if v["do-not-use"]}
        num_int = do_not_use.intersection(answers_dict.keys())
        print(
            f"Removing {num_int} answers whose associated questions were marked as DO_NOT_USE"
        )
        answers_dict = {k: v for k, v in answers_dict.items() if k not in do_not_use}
    save_to_jsonl(output_jsonl, answers_dict)


@main.command()
@click.argument("targets_jsonl", type=click.Path())
@click.argument("predictions_jsonl", type=click.Path())
@click.option(
    "--similarity_function",
    default="wordnet",
    type=click.Choice(["exact_match", "wordnet"], case_sensitive=False),
)
def evaluate(targets_jsonl, predictions_jsonl, similarity_function):
    """Run all evaluation metrics on model outputs"""
    from .data_processing import load_data_from_jsonl, load_predictions_from_jsonl
    from .evaluation import multiple_evals, all_eval_funcs

    print(f"Using {similarity_function} similarity.", flush=True)
    targets = load_data_from_jsonl(targets_jsonl)
    predictions = load_predictions_from_jsonl(predictions_jsonl)
    multiple_evals(
        eval_func_dict=all_eval_funcs[similarity_function],
        question_data=targets,
        answers_dict=predictions,
    )
