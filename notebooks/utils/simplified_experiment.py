import csv
import os

import pandas as pd
from fuzzywuzzy import fuzz

from notebooks.utils.process_lease_v2 import process_lease_v2


class SimplifiedExperiment:
    def __init__(
        self,
        answers_df,
        question_id,
        index_name: str,
        threshold=80,
        already_stored=True,
        already_vectorized=True,
        csv_results_filename="results.csv",
        lessor_question=False,
        date_question=False,
        boolean_question=False,
        template_string: str | None = None,
        retriever_question: str | None = None,
        use_llm_filter=False,
        use_listwise_rerank=False,
        use_embeddings_filter=False,
        temperature=0.0,
        chunks_filename: str | None = None,
    ):
        self.answers_df = answers_df
        self.question_id = question_id
        self.threshold = threshold
        self.already_stored = already_stored
        self.already_vectorized = already_vectorized
        self.index_name = index_name
        self.lessor_question = lessor_question
        self.numbers_list = [str(i).zfill(3) for i in range(1, 101)]
        self.total_answers = 0
        self.correct_answers = 0
        self.results = []
        self.csv_filename = csv_results_filename
        self.lease_number = 0
        self.date_question = date_question
        self.boolean_question = boolean_question
        self.template_string = template_string
        self.retriever_question = retriever_question
        self.use_llm_filter = use_llm_filter
        self.use_listwise_rerank = use_listwise_rerank
        self.use_embeddings_filter = use_embeddings_filter
        self.temperature = temperature
        self.chunks_filename = chunks_filename
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
        if self.chunks_filename:
            os.makedirs(os.path.dirname(self.chunks_filename), exist_ok=True)
            with open(self.chunks_filename, "w"):
                pass

    def run(self):
        with open(self.csv_filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Lease", "Answer", "Real answer", "Correct"])

        for number in self.numbers_list:
            file_name = f"lease{number}"
            self.file_name = file_name
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
            self.lease_number = int(number)

            try:
                result = process_lease_v2(
                    file_name=file_name,
                    namespace=namespace,
                    question_id=self.question_id,
                    answers_df=self.answers_df,
                    index_name=self.index_name,
                    template_string=self.template_string,
                    retriever_question=self.retriever_question,
                    use_llm_filter=self.use_llm_filter,
                    lessor_question=self.lessor_question,
                    date_question=self.date_question,
                    boolean_question=self.boolean_question,
                    temperature=self.temperature,
                    use_listwise_rerank=self.use_listwise_rerank,
                    use_embeddings_filter=self.use_embeddings_filter,
                    chunks_filename=self.chunks_filename,
                    lease_number=self.lease_number,
                )
                answer = result["answer"]
                real_answer_unprocessed = result["real_answer_unprocessed"]
                formatted_answer = real_answer_unprocessed

                if self.date_question:
                    formatted_answer = formatted_answer.strftime("%Y-%m-%d")

                self.evaluate_structured_output(answer, formatted_answer)
                print("----------------------------------------------------")
            except Exception as error:
                print(f"Error processing {file_name}: {error}")
                continue
        accuracy = self.correct_answers / self.total_answers
        print(f"Accuracy: {accuracy}")
        print(f"Results saved to {self.csv_filename}")

    def evaluate_structured_output(self, answer, formatted_answer):
        if self.lessor_question:
            extracted_answer = answer.answer_string
            self.evaluate_lessor_questions(extracted_answer, formatted_answer)
        elif self.date_question:
            extracted_answer = answer.answer_date
            self.evaluate_date_question(extracted_answer, formatted_answer)
        elif self.boolean_question:
            extracted_answer = answer.answer_boolean
            self.evaluate_boolean_question(extracted_answer, formatted_answer)

    def evaluate_lessor_questions(
        self,
        answer,
        formatted_answer,
    ):
        print(f"Lease: {self.file_name}")
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
                [self.lease_number, answer, formatted_answer, result_status]
            )

    def evaluate_date_question(self, answer, formatted_answer):
        print(f"Lease: {self.file_name}")
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
                [self.lease_number, answer, formatted_answer, result_status]
            )

    def evaluate_boolean_question(self, answer, formatted_answer):
        print(f"Lease: {self.file_name}")
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
                [self.lease_number, answer, formatted_answer, result_status]
            )
