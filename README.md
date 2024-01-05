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
                  """ ```


Los LLMs tienen ciertas limitaciones. Una de las limitaciones principales son las **Alucionaciones**. Esto quiere decir que el modelo puede dar respuestas que suenan plausibles pero realmente no son ciertas. Esto podría pasar por ejemplo pidiéndole la descripción de un artículo inventado:

```python3
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
```

### **3.Iterative Prompt Development**
En esta sección se enseña como analizar y refinar iterativamente los prompts para generar una copia de marketing del *fact sheet* de un producto.
El ciclo iterativo sería el siguiente:
* Realizar un prompt claro y específico
* Analizar por que el resultado no da el output deseado
* Refinar la idea y el prompt
* Repetir

Vamos a realizar el ejercicio:

En primer lugar, introducimos el texto con el *fact sheet* de una silla:

```python3
fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, 
including filing cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100) 
or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, 
gloss white, or chrome.
- Chair is available with or without armrests.
- Suitable for home or business settings.
- Qualified for contract use.

CONSTRUCTION
- 5-wheel plastic coated aluminum base.
- Pneumatic chair adjust for easy raise/lower action.

DIMENSIONS
- WIDTH 53 CM | 20.87”
- DEPTH 51 CM | 20.08”
- HEIGHT 80 CM | 31.50”
- SEAT HEIGHT 44 CM | 17.32”
- SEAT DEPTH 41 CM | 16.14”

OPTIONS
- Soft or hard-floor caster options.
- Two choices of seat foam densities: 
 medium (1.8 lb/ft3) or high (2.8 lb/ft3)
- Armless or 8 position PU armrests 

MATERIALS
SHELL BASE GLIDER
- Cast Aluminum with modified nylon PA6/PA66 coating.
- Shell thickness: 10 mm.
SEAT
- HD36 foam

COUNTRY OF ORIGIN
- Italy
"""
```

Ahora, generemos un primer promt para crear una campaña de marketing de la silla:

```python3
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```
```python3
Introducing our stunning mid-century inspired office chair, the perfect addition to any home or business setting. This chair is part of a beautiful family of office furniture, including filing cabinets, desks, bookcases, meeting tables, and more, all designed with a timeless mid-century aesthetic.

One of the standout features of this chair is the variety of customization options available. You can choose from several shell colors and base finishes to perfectly match your existing decor. The chair is available with either plastic back and front upholstery or full upholstery in a range of 10 fabric and 6 leather options, allowing you to create a look that is uniquely yours.

The chair is also available with or without armrests, giving you the flexibility to choose the option that best suits your needs. The base finish options include stainless steel, matte black, gloss white, or chrome, ensuring that you can find the perfect match for your space.

In terms of construction, this chair is built to last. It features a 5-wheel plastic coated aluminum base, providing stability and mobility. The pneumatic chair adjust allows for easy raise and lower action, ensuring optimal comfort throughout the day.

When it comes to dimensions, this chair is designed with both style and comfort in mind. With a width of 53 cm (20.87"), depth of 51 cm (20.08"), and height of 80 cm (31.50"), it offers ample space without overwhelming your space. The seat height is 44 cm (17.32") and the seat depth is 41 cm (16.14"), providing a comfortable seating experience for users of all heights.

We understand that every space is unique, which is why we offer a range of options to further customize your chair. You can choose between soft or hard-floor caster options, ensuring that your chair glides smoothly across any surface. Additionally, you have the choice between two seat foam densities: medium (1.8 lb/ft3) or high (2.8 lb/ft3), allowing you to select the level of support that suits your preferences. The chair is also available with armless design or 8 position PU armrests, providing additional comfort and versatility.

When it comes to materials, this chair is crafted with the utmost attention to quality. The shell base glider is made from cast aluminum with a modified nylon PA6/PA66 coating, ensuring durability and longevity. The shell thickness is 10 mm, providing a sturdy and reliable structure. The seat is made from HD36 foam, offering a comfortable and supportive seating experience.

Finally, this chair is proudly made in Italy, known for its exceptional craftsmanship and attention to detail. With its timeless design and superior construction, this chair is not only a stylish addition to any space but also a reliable and functional piece of furniture.

Upgrade your office or home with our mid-century inspired office chair and experience the perfect blend of style, comfort, and functionality.
```

**Problema 1: El texto de respuesta es demasiado largo**
Podemos limitar el número de palabras que queremos que el modelo genere:

```python3
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```
La nueva respuesta es la siguiente:
```python3
Introducing our mid-century inspired office chair, part of a stunning furniture collection. With various color and finish options, choose between plastic or full upholstery in fabric or leather. The chair features a durable aluminum base with 5 wheels and pneumatic height adjustment. Perfect for home or business use. Made in Italy.
```

**Problema 2: El texto se centra en los detalles incorrectos**
Una solución podría ser decirle al modelo que se centre en los detalles relevantes para la audiencia

```python3
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```
A lo que la nueva respuesta será:
```python3
Introducing our mid-century inspired office chair, perfect for both home and business settings. This chair is part of a beautiful family of furniture, offering a range of options including filing cabinets, desks, bookcases, and meeting tables. The chair is available with plastic or full upholstery in a variety of fabric and leather options. Choose from stainless steel, matte black, gloss white, or chrome base finishes. With its 5-wheel plastic coated aluminum base and pneumatic chair adjust, this chair offers easy raise/lower action. It also comes with the option of soft or hard-floor casters and two choices of seat foam densities. The chair is constructed with a cast aluminum shell with a modified nylon coating and a 10mm shell thickness. The seat is made with HD36 foam for maximum comfort. Made in Italy, this chair is qualified for contract use. Product ID: SWC-100, SWC-110.
```

**Problema 3: La descripción necesita una tabla**
Podemos pedirle al modelo que extraiga información y la organice en una tabla:

```python3
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

After the description, include a table that gives the 
product's dimensions. The table should have two columns.
In the first column include the name of the dimension. 
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. 
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)
```
A lo que el modelo nos devolverá el HTML:
```html
<div>
  <h2>Product Description</h2>
  <p>
    Introducing our latest addition to our mid-century inspired office furniture collection - the SWC Chair. This chair is part of a beautiful family of furniture that includes filing cabinets, desks, bookcases, meeting tables, and more. With its sleek design and customizable options, the SWC Chair is the perfect choice for any home or business setting.
  </p>
  <p>
    The SWC Chair is available in several options of shell color and base finishes, allowing you to create a look that matches your style. You can choose between plastic back and front upholstery or full upholstery in a variety of fabric and leather options. The base finish options include stainless steel, matte black, gloss white, or chrome. Additionally, you have the choice of having the chair with or without armrests.
  </p>
  <p>
    Constructed with durability in mind, the SWC Chair features a 5-wheel plastic coated aluminum base, ensuring stability and easy mobility. The chair also has a pneumatic adjuster, allowing for easy raise and lower action to find the perfect height for your comfort.
  </p>
  <p>
    The SWC Chair is not only stylish and functional, but it is also designed with your comfort in mind. The seat is made with HD36 foam, providing a comfortable and supportive seating experience. You also have the option to choose between soft or hard-floor caster options, depending on your flooring needs. Additionally, you can select between two choices of seat foam densities - medium (1.8 lb/ft3) or high (2.8 lb/ft3). The chair is also available with armless design or 8 position PU armrests for added convenience.
  </p>
  <p>
    Made with high-quality materials, the SWC Chair is built to last. The shell, base, and glider are constructed with cast aluminum with a modified nylon PA6/PA66 coating, ensuring durability and longevity. The shell has a thickness of 10 mm, providing stability and support. The chair is proudly made in Italy, known for its craftsmanship and attention to detail.
  </p>
  <p>
    The SWC Chair is not only a stylish addition to any space, but it is also a practical choice for both home and business settings. With its customizable options, durable construction, and comfortable design, the SWC Chair is the perfect seating solution for any environment.
  </p>
  <h2>Product Dimensions</h2>
  <table>
    <tr>
      <th>Dimension</th>
      <th>Measurement (inches)</th>
    </tr>
    <tr>
      <td>Width</td>
      <td>20.87"</td>
    </tr>
    <tr>
      <td>Depth</td>
      <td>20.08"</td>
    </tr>
    <tr>
      <td>Height</td>
      <td>31.50"</td>
    </tr>
    <tr>
      <td>Seat Height</td>
      <td>17.32"</td>
    </tr>
    <tr>
      <td>Seat Depth</td>
      <td>16.14"</td>
    </tr>
  </table>
</div>

Product IDs: SWC-100, SWC-110
```

