# ChatGPT Prompt Engineering for Developers
## Repositorio del curso 'ChatGPT Prompt Engineering for Developers' impartido por DeepLearning.AI
## Autor: Borja Esteve

### **1.Introducción**

Existen dos tipos de 'Large Language Models' (LLMs):
* LLM base: predice la siguiente palabra basándose en el texto de entrenamiento
* *Instruction Tuned LLM*:
    * Intenta seguir las instrucciones del usuario
    * Es un modelo base al que se le ha realizado *fine tunning* sobre instrucciones y buenas formas de seguir esas instrucciones
    * Usa *Reinforcement Learning with Human Feedback* 
Este curso se centra en los *Instruction Tuned LLM*

### **2.Guidelines for Prompting**
Principios del prompting:
* Principio 1:Escribe instrucciones claras y específicas. Algunas tácticas son:
    * El uso de delimitadores como: triple """, triple ```, triple ---, <> o XML tags.
        * Ejemplo: `prompt = f""" Summarize the text delimited by triple backticks \ into a single sentence. ```{text}``` """`
    * Preguntar por outputs estructurados: Por ejemplo, pedirle al modelo que lo estructure en un json o HTML.
    * Preguntar al modelo que chequee si se cumplen unas condiciones o comprobar los supuestos necesarios para realizar la tarea.
        * Ejemplo: ```python3
                    prompt = f"""
                    You will be provided with text delimited by triple quotes. 
                    If it contains a sequence of instructions, \ 
                    re-write those instructions in the following format:

                    Step 1 - ...
                    Step 2 - …
                    …
                    Step N - …

                    If the text does not contain a sequence of instructions, \ 
                    then simply write \"No steps provided.\"
                    
                    \"\"\"{text_1}\"\"\"
                    """
                    ```
* Principio 2:
    * Dale al modelo tiempo para pensar
Para este apartado se usa la librería `openai`de `Python`. 

```python3
import openai
openai.api_key = "CLAVE DE LA API DE OPENAI"
```

Después, creamos una función la cual recibirá el promp y el tipo de modelo a usar y devolverá la respuesta al prompt. En este caso usaremos el modelo `gpt-3.5-turbo`.
```python3
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # Grados de aleatoriedad del modelo
    )
    return response.choices[0].message["content"]
```


