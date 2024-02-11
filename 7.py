import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from janome.tokenizer import Tokenizer

class DialogueSystem:
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, is_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.janome_tokenizer = Tokenizer()
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.latest_pronouns = {'1st': None, '2nd': None, '3rd': None}

    def extract_and_update_pronouns(self, text):
        tokens = self.janome_tokenizer.tokenize(text)
        for token in tokens:
            pronoun_type = self.classify_pronoun(token.surface)
            if pronoun_type:
                self.latest_pronouns[pronoun_type] = token.surface

    def classify_pronoun(self, pronoun):
        first_person_pronouns = ["私", "僕", "俺", "わたし", "わたくし", "あたし", "おれ", "わし", "わい", "吾輩"]
        second_person_pronouns = ["あなた", "君", "貴方", "貴女"]
        third_person_pronouns = ["彼", "彼女", "あいつ", "彼ら"]
        if pronoun in first_person_pronouns:
            return '1st'
        elif pronoun in second_person_pronouns:
            return '2nd'
        elif pronoun in third_person_pronouns:
            return '3rd'
        return None

    def generate_response(self, input_text):
        self.extract_and_update_pronouns(input_text)

        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        if self.use_gpu:
            input_ids = input_ids.cuda()

        output_ids = self.model.generate(input_ids, max_length=512, temperature=1.0, repetition_penalty=1.5)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        response = self.replace_pronouns_in_response(response)
        return response

    def replace_pronouns_in_response(self, response):
        tokens = self.janome_tokenizer.tokenize(response)
        updated_response = ''
        for token in tokens:
            pronoun_type = self.classify_pronoun(token.surface)
            if pronoun_type and self.latest_pronouns[pronoun_type]:
                updated_response += self.latest_pronouns[pronoun_type]
            else:
                updated_response += token.surface
        return updated_response

model_path = 'D:\GitHub\chatbot\model_1'
dialogue_system = DialogueSystem(model_path)

# 运行对话系统
while True:
    user_input = input("user: ")
    if user_input.lower() == 'exit':
        break
    response = dialogue_system.generate_response(user_input)
    print("bot: ", response)
