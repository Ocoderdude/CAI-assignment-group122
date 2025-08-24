#!/usr/bin/env python3
"""
Simple test to check if the FLAN-T5-base model is working correctly.
"""

def test_model_loading():
    print("=== Testing FLAN-T5-base Model ===\n")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        print("✅ Tokenizer loaded successfully")
        
        print("\n2. Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        print("✅ Model loaded successfully")
        
        print("\n3. Testing basic generation...")
        model.eval()
        
        # Simple test prompt
        test_prompt = "Question: What is 2+2? Answer:"
        inputs = tokenizer(test_prompt, return_tensors="pt", max_length=50, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, num_beams=3, do_sample=False)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test generation successful: '{answer}'")
        
        print("\n4. Testing financial prompt...")
        financial_prompt = "Question: What was the revenue? Answer:"
        inputs = tokenizer(financial_prompt, return_tensors="pt", max_length=100, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, num_beams=3, do_sample=False)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Financial generation successful: '{answer}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model is working correctly!")
    else:
        print("\n💥 Model has issues that need fixing.")
