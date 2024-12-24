import os
import random
import json
import nltk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Chatbot:
    def __init__(self, intents_file=None):
        if intents_file is None:
            intents_file = os.path.join(os.path.dirname(__file__), 'intents.json')
        with open(intents_file, 'r', encoding='utf-8') as file:
            self.intents = json.load(file)

    def preprocess_input(self, user_input):
        nltk.download('punkt', quiet=True)
        return nltk.word_tokenize(user_input.lower().strip())

    def get_exchange_rates(self, base_currency="USD", target_currencies=None):
        if target_currencies is None:
            target_currencies = ["TRY", "EUR", "GBP", "JPY"]
        api_key = "6f681993510e345ffe0d4c81"
        rates = []
        for target_currency in target_currencies:
            base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                rate = data["conversion_rate"]
                rates.append(f"1 {base_currency} = {rate} {target_currency}")
            else:
                rates.append(f"Döviz kuru bilgisi alınamadı: {base_currency} -> {target_currency}")
        return "\n".join(rates)

    def get_weather(self, city="Istanbul"):
        api_key = "6adc8773a6f99986033cde5a6634c39e"  
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city + ",TR",
            "appid": api_key,
            "units": "metric",
            "lang": "tr"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            description = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            return f"{city} için hava durumu: {description.capitalize()}, sıcaklık: {temp}°C, hissedilen sıcaklık: {feels_like}°C"
        elif response.status_code == 404:
            return f"{city} şehri için hava durumu bilgisi bulunamadı. Lütfen doğru bir şehir adı yazdığınızdan emin olun."
        else:
            return f"Hava durumu bilgisi alınamadı. Hata kodu: {response.status_code}"

    def get_response(self, user_input):
        user_input = user_input.lower().strip()

        if "hava" in user_input or "hava durumu" in user_input:
            city = "Istanbul"  
            if "hava durumu" in user_input:
                city = user_input.replace("hava durumu", "").strip()
            elif "hava" in user_input:
                city = user_input.replace("hava", "").strip()
            return self.get_weather(city)

        if "döviz" in user_input or "dolar" in user_input:
            base_currency = "USD"
            if "euro" in user_input:
                base_currency = "EUR"
            return self.get_exchange_rates(base_currency)

        qa_patterns = []
        qa_responses = []
        for intent in self.intents['intents']:
            for qa_pair in intent.get('qa_pairs', []):
                qa_patterns.append(qa_pair['pattern'].lower())
                qa_responses.append(qa_pair['response'])

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(qa_patterns + [user_input])
        user_vector = vectors[-1]
        qa_vectors = vectors[:-1]

        similarities = cosine_similarity(user_vector, qa_vectors).flatten()
        max_similarity_index = similarities.argmax()

        if similarities[max_similarity_index] > 0.5:
            return qa_responses[max_similarity_index]

        patterns = []
        intent_tags = []
        for intent in self.intents['intents']:
            patterns.extend(intent.get('patterns', []))
            intent_tags.extend([intent['tag']] * len(intent.get('patterns', [])))

        vectors = vectorizer.fit_transform(patterns + [user_input])
        user_vector = vectors[-1]
        pattern_vectors = vectors[:-1]
        similarities = cosine_similarity(user_vector, pattern_vectors).flatten()
        max_similarity_index = similarities.argmax()

        if similarities[max_similarity_index] > 0.4:
            matched_tag = intent_tags[max_similarity_index]
            for intent in self.intents['intents']:
                if intent['tag'] == matched_tag:
                    return random.choice(intent.get('responses', []))

        return "Üzgünüm, anlamadım. Daha net bir şey sorabilir misiniz?"
