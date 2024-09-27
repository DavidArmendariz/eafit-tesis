import csv
import os

import pandas as pd
from fuzzywuzzy import fuzz

from openai_pinecone.services.upload_to_s3 import upload_to_s3
from utils.constants import AvailableEmbeddingModels, AvailableModels
from utils.process_lease import process_lease
from utils.storage import store_pdf_in_vector_db


class DocumentsPreprocessing:
    def __init__(
        self,
        index_name: str,
        embedding_model: AvailableEmbeddingModels,
    ):
        self.numbers_list = [str(i).zfill(3) for i in range(1, 101)]
        self.index_name = index_name
        self.embedding_model: AvailableEmbeddingModels = embedding_model

    def store_in_s3(self):
        for number in self.numbers_list:
            file_name = f"lease{number}"
            local_file = f"../data/asc_842/lease_agreements/{file_name}.pdf"
            file_key = f"eafit/{file_name}.pdf"
            upload_to_s3(local_file, file_key)

    def store_in_vector_db(self):
        for number in self.numbers_list:
            file_name = f"lease{number}"
            file_key = f"eafit/{file_name}.pdf"
            namespace = f"eafit_{file_name}"
            store_pdf_in_vector_db(
                file_key,
                namespace,
                embedding_model=self.embedding_model,
                index_name=self.index_name,
            )


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
        self.numbers_list = [str(i).zfill(3) for i in range(1, 101)]
        self.total_answers = 0
        self.correct_answers = 0
        self.results = []
        self.csv_filename = csv_results_filename
        self.lease_number = 0
        self.date_question = date_question
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
            self.evaluate_lessor_questions(answer.answer_string[0], formatted_answer)
        else:
            if self.date_question:
                extracted_answer = answer.answer_date
            self.evaluate_exact_match(extracted_answer, formatted_answer)

    def evaluate_unstructured_output(self, answer, formatted_answer):
        if self.lessor_question:
            self.evaluate_lessor_questions(answer[0], formatted_answer)
        else:
            self.evaluate_exact_match(answer, formatted_answer)

    def evaluate_exact_match(self, answer, formatted_answer):
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
