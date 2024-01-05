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
    * El **uso de delimitadores** como: triple """, triple ```, triple ---, <> o XML tags.
        * Ejemplo: `prompt = f""" Summarize the text delimited by triple backticks \ into a single sentence. ```{text}``` """`
    * Preguntar por **outputs estructurados**: Por ejemplo, pedirle al modelo que lo estructure en un json o HTML.
    * Preguntar al modelo que chequee si **se cumplen unas condiciones** o comprobar los **supuestos necesarios para realizar la tarea**.
        * Ejemplo: ```python3
                    prompt = f"""
                    You will be provided with text delimited by triple quotes. 
                    If it contains a sequence of instructions, \ 
                    re-write those instructions in the following format:
                    Step 1 - ...
                    Step 2 - ...
                    Step N - ...
                    If the text does not contain a sequence of instructions, \ 
                    then simply write \"No steps provided.\"
                    \"\"\"{text_1}\"\"\"
                    """
                    ```
    * **Few-shot prompting**: Se le dan ejemplos exitosos de ejecuciones de la tarea que desea realizar antes de pedirle al modelo que realice la tarea real                               que desea que realice. 
* Principio 2: Dale al modelo tiempo para pensar. Cuando el promopt es muy complejo, puedes instruir al modelo para que piense durante más tiempo acerca de                 la respuesta.
    * **Especificar los pasos en los que completar la tarea**:
        * ```python3
             prompt_1 = f"""
            Perform the following actions: 
            1 - Summarize the following text delimited by triple \
            backticks with 1 sentence.
            2 - Translate the summary into French.
            3 - List each name in the French summary.
            4 - Output a json object that contains the following \
            keys: french_summary, num_names.
            
            Separate your answers with line breaks.
            
            Text:
            ```{text}```
            """ ```
    * **Indique al modelo que encuentre su propia solución antes de apresurarse a llegar a una conclusión.**:
       * Ejemplo:
               ```python3
                  prompt = f"""
                  Determine if the student's solution is correct or not.
                  Question:
                  I'm building a solar power installation and I need \
                   help working out the financials. 
                  - Land costs $100 / square foot
                  - I can buy solar panels for $250 / square foot
                  - I negotiated a contract for maintenance that will cost \ 
                  me a flat $100k per year, and an additional $10 / square \
                  foot
                  What is the total cost for the first year of operations 
                  as a function of the number of square feet.
                  Student's Solution:
                  Let x be the size of the installation in square feet.
                  Costs:
                  1. Land cost: 100x
                  2. Solar panel cost: 250x
                  3. Maintenance cost: 100,000 + 100x
                  Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
                  """
            ```

          
  
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


