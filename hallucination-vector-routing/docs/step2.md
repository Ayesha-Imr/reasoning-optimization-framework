### **Documentation: Completion of Step 2 - Risk Scorer Training and Validation**

#### **Objective**
To transform the `v_halluc` persona vector (from Step 1) into a practical, real-time risk predictor. This involved creating a large, labeled dataset to train a logistic regression model capable of assigning a calibrated hallucination probability (0.0 to 1.0) to any given prompt based on its projection onto the vector.

#### **Outcome**
This step has been successfully completed. The primary artifacts—a labeled dataset, a trained classifier, and a real-time risk scoring function—have been generated and validated. The classifier's performance **exceeded the project's target AUROC of ≥0.75**.

---

#### **Summary of Process and Methodology**

The process was executed in a single Google Colab notebook and was broken down into four distinct phases:

**Phase 1: Multi-Scenario Dataset Generation**
To address the natural class imbalance where the base model rarely hallucinates, a multi-scenario data generation strategy was implemented using the SQuAD dataset. 2,000 prompts were processed in three ways to induce a wider range of behaviors:
*   **Standard (1000 examples):** The model was given the correct context and question. This produced mostly faithful answers (label `0`).
*   **No-Context (500 examples):** The model was given the question without context, forcing it to rely on its parametric memory and increasing the likelihood of natural hallucination.
*   **Distractor-Context (500 examples):** The model was given the question with an incorrect, unrelated context, testing its ability to recognize when an answer is not present.

All 2,000 generated answers were judged by a Gemini 1.5 Flash-based LLM judge, which compared the model's answer to the ground-truth context to assign a binary `hallucination_label`. This resulted in a high-quality, balanced dataset.

**Phase 2: Feature Calculation**
A `z_feature` was computed for every prompt in the newly created dataset. This was achieved by:
1.  Running each prompt through the Llama 3 model to get its hidden states.
2.  Extracting the activation vector of the **last prompt token** at the pre-selected `TARGET_LAYER = 16`.
3.  Calculating the dot product (projection) of this activation with the `v_halluc` vector.
The process was executed in a resilient, batched loop, saving the final DataFrame with the computed features.

**Phase 3: Classifier Training and Validation**
The dataset was split into an 80% training set and a 20% test set, stratified to maintain the class balance. A `scikit-learn` Logistic Regression model was trained on the `(z_feature, hallucination_label)` pairs. The model was then rigorously evaluated on the held-out test set.

**Phase 4: Real-Time Risk Function Creation**
All logic from Phases 2 and 3 was encapsulated into a single, efficient Python function, `get_hallucination_risk`, which takes a raw prompt and returns a final, calibrated risk score.

---

#### **Key Artifacts and Constants**

*   **Primary Artifacts:**
    *   **File:** `risk_clf.joblib`
        *   **Description:** The saved, trained `scikit-learn` Logistic Regression classifier.
    *   **File:** `squad_data_with_features.csv`
        *   **Description:** The final dataset containing the prompts, labels, and the computed `z_feature` for all 2,000 examples.

*   **Key Constants Used:**
    *   **Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
    *   **Source Dataset:** `rajpurkar/squad`
    *   **Sample Size:** `N_SAMPLES = 2000`
    *   **Target Layer:** `TARGET_LAYER = 16`
    *   **Judge Model:** `gemini-2.5-flasht`

---

#### **Defined Functions**

*   `generate_squad_answer_multi_scenario(model, tokenizer, row)`: Generated a model answer based on the specified scenario (standard, no-context, or distractor).
*   `judge_squad_answer(...)`: Queried the Gemini API to get a binary `0` or `1` hallucination label for a generated answer.
*   `get_last_prompt_token_activation(model, tokenizer, prompt_text)`: Extracted the Layer 16 activation vector for the last token of a given prompt.
*   `get_hallucination_risk(prompt_text, ...)`: The final, real-time function that encapsulates the entire process from prompt to risk score.

---

#### **Key Results and Validation**

The risk scorer's performance on the held-out test set was excellent and successfully met the project's target.

*   **Primary Metric - AUROC:**
    *   **Score:** **0.8622**
    *   **Outcome:** ✅ **Success.** This result significantly surpasses the target of ≥0.75, confirming that the projection onto `v_halluc` is a strong predictor of hallucination.

*   **Classification Report (at 0.5 Threshold):**
    *   **Precision (for Hallucination):** **0.82** — When the model predicts a prompt is risky, it is correct 82% of the time.
    *   **Recall (for Hallucination):** **0.61** — The model successfully identifies 61% of all actual hallucinations at this threshold.
    *   This demonstrates a strong precision-recall balance, which can be tuned later by adjusting the risk thresholds in our guardrail.

*   **Real-Time Function Test:**
    *   **Risk for a Safe Prompt:** `0.1650` (Low Risk)
    *   **Risk for a Risky Prompt:** `0.8009` (High Risk)
    *   The clear separation in scores confirms the function is working as expected.

This completes the documentation for Step 2. We have successfully built and validated a high-quality, real-time risk scorer, which will serve as the core decision-making engine for the three-tier guardrail in Step 3.