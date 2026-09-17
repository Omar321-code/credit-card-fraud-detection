import gradio as gr
import joblib
import pandas as pd
import numpy as np

try: 
    model = joblib.load("Model/model.joblib")
except Exception as e:    
    model = None


def predict_fraud (*all_inputs):

  if model is None :
   return  "⚠️ Model file not found yet!"

  time = all_inputs[0]
  Vfeatures = all_inputs[1:29]
  amount = all_inputs[29]
  feature_values = list(all_inputs)

  feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
  input_df = pd.DataFrame([feature_values], columns=feature_names)
  
  prediction = model.predict(input_df)


  if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        prob_text = f" (Fraud Probability: {prob:.2%})"

  if prediction[0] == 1:
        return f"🚨 Fraudulent Transaction Detected!{prob_text}"
  else:
        return f"✅ Legitimate Transaction{prob_text}"
  
inputs = [gr.Number(value=0.0, label="Time (Seconds elapsed)")] + \
         [gr.Number(value=0.0, label=f"V{i}") for i in range(1, 29)] + \
         [gr.Number(value=0.0, label="Amount ($)")]
UI = gr.Interface(
    fn = predict_fraud,
    inputs = inputs,
    outputs = "text",
    title = "💳 Credit Card Fraud Detection System",
    description="Enter the transaction parameters below to evaluate whether the transaction is fraudulent or legitimate.",
    examples=[[50.0] + [1.2] * 28 + [200.0]]
)

# def predict_fraud(time, *v_features, amount):
#     if model is None:
#         return "⚠️ Model file not found in models/model.joblib!"
    
   
#     features = [time] + list(v_features) + [amount]
#     X = np.array([features])
    
#     prediction = model.predict(X)[0]
#     return "🚨 Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"


# with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # gr.Markdown("# 💳 Credit Card Fraud Detection System")
    # gr.Markdown("Enter transaction parameters below to evaluate legitimacy.")
    
    # with gr.Row():
    #     with gr.Column():
            
    #         time_input = gr.Number(label="Time (Seconds elapsed)", value=0)
            
           
    #         with gr.Accordion("🔒 Advanced PCA Features (V1 - V28)", open=False):
    #             v_inputs = []
    #             for i in range(1, 29):
    #                 v_inputs.append(gr.Number(label=f"V{i}", value=0.0))
            
    #         amount_input = gr.Number(label="Amount ($)", value=100.0)
    #         submit_btn = gr.Button("Evaluate Transaction", variant="primary")
            
    #     with gr.Column():
    #         output_box = gr.Textbox(label="Evaluation Result")


    # all_inputs = [time_input] + v_inputs + [amount_input]
    
    # submit_btn.click(
    #     fn=predict_fraud,
    #     inputs=all_inputs,
    #     outputs=output_box
    # )

if __name__ == "__main__":

    UI.launch()