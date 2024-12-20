import csv
import os

import pandas as pd
from fuzzywuzzy import fuzz

from .constants import AvailableEmbeddingModels, AvailableModels
from .process_lease import process_lease


class Experiment:
    def __init__(
        self,
        answers_df,
        question_id,
        question_for_ai,
        index_name: str,
        threshold=80,
        model: AvailableModels = "gpt-3.5-turbo",
        embedding_model: AvailableEmbeddingModels = "text-embedding-ada-002",
        already_stored=True,
        already_vectorized=True,
        use_structured_outputs=True,
        csv_results_filename="results.csv",
        lessor_question=False,
        date_question=False,
        boolean_question=False,
        template_string: str | None = None,
        retriever_question: str | None = None,
        numbers_list: list[str] | None = None,
    ):
        self.answers_df = answers_df
        self.question_id = question_id
        self.question_for_ai = question_for_ai
        self.threshold = threshold
        self.model: AvailableModels = model
        self.embedding_model: AvailableEmbeddingModels = embedding_model
        self.already_stored = already_stored
        self.already_vectorized = already_vectorized
        self.use_structured_outputs = use_structured_outputs
        self.index_name = index_name
        self.lessor_question = lessor_question
        self.numbers_list = numbers_list or [str(i).zfill(3) for i in range(1, 101)]
        self.total_answers = 0
        self.correct_answers = 0
        self.results = []
        self.csv_filename = csv_results_filename
        self.lease_number = 0
        self.date_question = date_question
        self.boolean_question = boolean_question
        self.template_string = template_string
        self.retriever_question = retriever_question
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)

    def run(self):
        with open(self.csv_filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Lease", "Answer", "Real answer", "Correct"])

        for number in self.numbers_list:
            file_name = f"lease{number}"
            namespace = f"eafit_{file_name}"

            # Check if the real answer is NaT
            answer_is_not_given = pd.isna(
                self.answers_df[self.answers_df["Lease"] == file_name][
                    self.question_id
                ].iloc[0]
            )

            if answer_is_not_given:
                continue

            self.total_answers += 1
            self.lease_number = number

            try:
                result = process_lease(
                    file_name=file_name,
                    namespace=namespace,
                    question_for_ai=self.question_for_ai,
                    question_id=self.question_id,
                    answers_df=self.answers_df,
                    model=self.model,
                    embedding_model=self.embedding_model,
                    use_structured_outputs=self.use_structured_outputs,
                    index_name=self.index_name,
                    template_string=self.template_string,
                    retriever_question=self.retriever_question,
                )
                answer = result["answer"]
                real_answer_unprocessed = result["real_answer_unprocessed"]
                formatted_answer = real_answer_unprocessed.iloc[0]

                if self.date_question:
                    formatted_answer = formatted_answer.strftime("%Y-%m-%d")

                if self.use_structured_outputs and self.model != "gpt-3.5-turbo":
                    self.evaluate_structured_output(answer, formatted_answer)
                else:
                    self.evaluate_unstructured_output(answer, formatted_answer)
                print("----------------------------------------------------")
            except Exception as error:
                print(f"Error processing {file_name}: {error}")
                continue
        accuracy = self.correct_answers / self.total_answers
        print(f"Accuracy: {accuracy}")
        print(f"Results saved to {self.csv_filename}")

    def evaluate_structured_output(self, answer, formatted_answer):
        if self.lessor_question:
            extracted_answer = answer.answer_string[0]
            self.evaluate_lessor_questions(extracted_answer, formatted_answer)
        elif self.date_question:
            extracted_answer = answer.answer_date
            self.evaluate_date_question(extracted_answer, formatted_answer)
        elif self.boolean_question:
            extracted_answer = answer.answer_boolean
            self.evaluate_boolean_question(extracted_answer, formatted_answer)

    def evaluate_unstructured_output(self, answer, formatted_answer):
        if self.lessor_question:
            self.evaluate_lessor_questions(answer[0], formatted_answer)
        elif self.date_question:
            self.evaluate_date_question(answer, formatted_answer)
        elif self.boolean_question:
            self.evaluate_boolean_question(answer, formatted_answer)

    def evaluate_boolean_question(self, answer, formatted_answer):
        print(f"Answer: {answer}")
        print(f"Real answer: {formatted_answer}")
        if int(answer) == int(formatted_answer):
            self.correct_answers += 1
            result_status = 1  # Correct
            print("CORRECT")
        else:
            result_status = 0  # Incorrect
            print("INCORRECT")
        with open(self.csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [int(self.lease_number), answer, formatted_answer, result_status]
            )

    def evaluate_date_question(self, answer, formatted_answer):
        print(f"Answer: {answer}")
        print(f"Real answer: {formatted_answer}")
        if answer == formatted_answer:
            self.correct_answers += 1
            result_status = 1  # Correct
            print("CORRECT")
        else:
            result_status = 0  # Incorrect
            print("INCORRECT")
        with open(self.csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [int(self.lease_number), answer, formatted_answer, result_status]
            )

    def evaluate_lessor_questions(
        self,
        answer,
        formatted_answer,
    ):
        print(f"Answer: {answer}")
        print(f"Real answer: {formatted_answer}")
        similarity_score = fuzz.ratio(answer.lower(), formatted_answer.lower())
        if similarity_score >= self.threshold:
            self.correct_answers += 1
            result_status = 1  # Correct
            print("CORRECT")
        else:
            result_status = 0  # Incorrect
            print("INCORRECT")
        with open(self.csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [int(self.lease_number), answer, formatted_answer, result_status]
            )
