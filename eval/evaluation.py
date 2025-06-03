from datasets import Dataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from eval.custom_llm import CustomLlama 

def evaluate_outfit_response(query, context, llm_output):
    print("\n Running evaluation for outfit generation using DeepEval (local model)...\n")

    # DeepEval using the registered local model
    test_case = LLMTestCase(
        input=query,
        actual_output=llm_output,
        context=[context]
    )

    custom_llm = CustomLlama()

    relevance = AnswerRelevancyMetric(model=custom_llm).measure(test_case)

    print("\n🧪 DeepEval Results:")
    print(f"• Relevance:   {relevance:.2f}")
