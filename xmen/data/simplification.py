from xmen.log import logger
import datasets


class GPTSimplifier:
    def __init__(self, model=None, open_ai_api_key=None, fixed_few_shot_examples=[], table={}):
        self.table = table
        self.model = model
        if not model and not table:
            logger.warning("No model or lookup table provided, the simplifier will have no effect")
        if model:
            from openai import OpenAI

            self.prompt_template = self.get_prompt(fixed_few_shot_examples)
            self.client = OpenAI(api_key=open_ai_api_key)

    def get_prompt(self, few_shot_pairs):
        prompt = "Your task is to simplify expressions, such that the simplified version is more suitable for looking up concepts in a medical terminology. If the input is already simple enough, just return the input. \n\n"
        if len(few_shot_pairs) > 0:
            prompt += "Here are some examples:\n"
        for original, solution in few_shot_pairs:
            prompt += f"Input: ```{original}```\n"
            prompt += f"Simplified: ```{solution}```\n\n"

        prompt += "Input: ```{}```\n"
        prompt += f"Simplified:"
        return prompt

    # Helper function to send messages to OpenAI API
    def get_completion(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content.replace("```", "")

    def simplify(self, text):
        if text in self.table:
            return self.table[text].lower()
        elif self.model:
            logger.debug(f"Calling {self.model} with mention: {text}")
            result = self.get_completion(self.prompt_template.format(text)).lower()
            self.table[text] = result
            return result
        else:
            return text


class EntitySimplification:
    def __getstate__(self):
        return {}

    @staticmethod
    def filter_n(n):
        return lambda e: len(" ".join(e["text"]).split(" ")) >= n

    def __init__(self, simplifier, filter_fn=None, set_long_form=False):
        self.simplifier = simplifier
        if not filter_fn:
            filter_fn = lambda _: True
        self.filter_fn = filter_fn
        self.set_long_form = set_long_form

    def _simplify_entities(self, d):
        ents = []
        for e in d["entities"]:
            assert len(e["text"]) == 1
            if self.filter_fn(e):
                if self.set_long_form:
                    e["long_form"] = e["text"][0]
                e["text"] = [self.simplifier.simplify(e["text"][0])]
            ents.append(e)
        return {"entities": ents}

    def transform_batch(self, ds):
        return ds.map(self._simplify_entities, load_from_cache_file=False)


class SimplifierWrapper:
    def __getstate__(self):
        return {}

    def __init__(self, linker, text_simplifier, filter_fn, set_long_form=True):
        self.linker = linker
        self.simplifier = EntitySimplification(text_simplifier, filter_fn=filter_fn, set_long_form=set_long_form)

    def predict_batch(self, dataset, **linker_kwargs):
        simplified_ds = self.simplifier.transform_batch(dataset)
        simplified_candidates = self.linker.predict_batch(simplified_ds, **linker_kwargs)

        def reset_entities(d, i, ds):
            # Reset text and long_form, because these have been messed with by the Simplifier component
            original_doc = ds[i]
            entities = []
            for e, eo in zip(d["entities"], original_doc["entities"]):
                e["text"] = eo["text"]
                e["long_form"] = eo["long_form"]
                entities.append(e)

            return {"entities": entities}

        if type(dataset) == datasets.DatasetDict:
            result = datasets.DatasetDict()
            for k, v in dataset.items():
                result[k] = simplified_candidates[k].map(lambda d, i: reset_entities(d, i, v), with_indices=True)
            return result
        else:
            result = simplified_candidates.map(lambda d, i: reset_entities(d, i, dataset), with_indices=True)
            return result
