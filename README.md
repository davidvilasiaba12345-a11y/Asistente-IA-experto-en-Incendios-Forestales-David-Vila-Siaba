# 🔥 Vigía Forestal - Asistente Experto en Incendios Forestales en España

**Asistente conversacional con Google Gemini, RAG y LangGraph

Proyecto final del módulo de IA Generativa · Máster en Data Science & IA · Evolve
Autor: **David Vila Siaba**

---

## 📌 Descripción del proyecto

Vigía Forestal es un asistente de inteligencia artificial especializado en **incendios forestales en España**. Permite hacer preguntas en lenguaje natural y obtener respuestas precisas, fundamentadas en documentación oficial y en datos históricos, indicando siempre la fuente de cada información.

El asistente combina tres técnicas de IA Generativa:

- **RAG (Retrieval-Augmented Generation):** antes de responder, recupera de una base de conocimiento vectorial (ChromaDB) los fragmentos más relevantes para la pregunta y los utiliza como contexto, lo que reduce las alucinanciones y permite citar las fuentes.
- **Agente en LangGraph:** el flujo de cada pregunta se modela como un grafo de tres nodos (reformular, recuperar y generar) que comparten un estado común.
- **Memoria de conversación:** el asistente recuerda los turnos anteriores, por lo que admite preguntas de seguimiento como *"¿y cuántas personas participaron en su extinción?"*.

El proyecto incluye un **notebook** en el que se construye y evalúa el asistente paso a paso, y una **aplicación web en Streamlit** desplegada en Streamlit Cloud para conversar con él desde el navegador.

| Recurso | Enlace |
|---|---|
| 📓 Notebook del proyecto | [`asistente_incendios.ipynb`](asistente_incendios.ipynb) |
| 💬 Aplicación web | [Vigía Forestal en Streamlit Cloud](https://vigia-forestal.streamlit.app/) |

---

## 🌲 Dominio elegido: incendios forestales en España

He elegido los incendios forestales en España porque son uno de los principales problemas medioambientales del país. Según los datos de la Estadística General de Incendios Forestales que utilizo en el proyecto, entre 1968 y 2022 se registraron **296.816 incendios de al menos 1 hectárea**, que quemaron en total **más de 8,1 millones de hectáreas**.

A pesar de su relevancia, la información sobre este tema está dispersa en informes técnicos extensos y en bases de datos poco accesibles para el público general. Mi objetivo es que cualquier persona pueda consultarla de forma sencilla, en lenguaje natural, y con la garantía de que cada respuesta procede de una fuente oficial.

El asistente puede responder a preguntas de distintos tipos:

| Tipo de pregunta | Ejemplo |
|---|---|
| Conceptos y metodología | ¿Qué diferencia hay entre un conato y un incendio forestal? |
| Evolución histórica | ¿Cuál fue el peor año de incendios en España? |
| Distribución territorial | ¿Qué provincia de Galicia ha tenido más superficie quemada? |
| Causas y motivaciones | ¿Por qué se provocan los incendios intencionados? |
| Grandes incendios | ¿Qué ocurrió en el incendio de Cortes de Pallás de 2012? |
| Prevención y autoprotección | ¿Qué hago si me sorprende un incendio en el monte? |

Además, el asistente reconoce los límites de su base de conocimiento: si no dispone de una información (por ejemplo, datos posteriores a 2022 o de otros países), lo indica en lugar de inventarla.

---

## 📚 Base de conocimiento

La base de conocimiento combina dos tipos de fuentes que se complementan: documentos PDF con información explicativa y un dataset histórico con los datos de cada incendio.

### Documentos PDF

| Documento | Fuente | Contenido |
|---|---|---|
| [Los Incendios Forestales en España. Decenio 2006-2015](https://www.miteco.gob.es/es/biodiversidad/temas/incendios-forestales/incendios-decenio-2006-2015_tcm30-521617.pdf) | Ministerio de Agricultura, Pesca y Alimentación | Evolución de los incendios, grandes incendios, causas, distribución territorial y temporal, extinción y prevención |
| [Metodología de la Estadística General de Incendios Forestales](https://www.miteco.gob.es/content/dam/miteco/es/biodiversidad/temas/incendios-forestales/estadística-iiff/Estadística-General-IF-Metodología.pdf) | MITECO | Definición de las variables estadísticas y de las categorías de causa |
| [Consejos ante Incendios Forestales](https://www.jerez.es/fileadmin/_migrated/content_uploads/Consejos_Incendios_Forestales.pdf) | Protección Civil, Ayuntamiento de Jerez | Recomendaciones de prevención y autoprotección |

### Dataset histórico

- **`fires-all.csv`:** 296.839 incendios forestales registrados en España entre 1968 y 2023, con superficie quemada, fecha, ubicación, causa, víctimas, personal de extinción y tiempos de extinción. Procede del conjunto de datos [*Todos los incendios forestales*](https://datos.civio.es/dataset/todos-los-incendios-forestales/), publicado por la Fundación Ciudadana Civio a partir de la Estadística General de Incendios Forestales (EGIF) del Ministerio de Agricultura, Pesca y Alimentación.
- **`eell-extra-codes.xlsx`:** [tablas de códigos adicionales](https://civio.app.box.com/s/6jcg8g9iku245jkrv2rmxfls3pknlpyt) publicadas por Civio junto al dataset, utilizadas para traducir los códigos de comunidad, provincia, causa y subcausa a nombres legibles.

### Preparación de los datos

| Fuente | Procesamiento | Documentos indexados |
|---|---|---|
| PDF | Limpieza del texto extraído (cabeceras, números de página, palabras partidas) y división en fragmentos de 1.500 caracteres con un solapamiento de 200 | 379 fragmentos |
| CSV | Limpieza y traducción de códigos con pandas, descarte del año 2023 por estar incompleto y generación de textos descriptivos: 1 resumen general, 55 resúmenes anuales, 18 resúmenes por comunidad autónoma y 180 fichas de los incendios de 3.000 hectáreas o más | 254 documentos |

El CSV no se indexa fila a fila: un registro numérico aislado apenas tiene significado para un modelo de lenguaje, y la búsqueda semántica solo recupera unos pocos documentos, por lo que nunca podría responder a preguntas que requieren agregar todo el dataset. Por eso lo transformo en resúmenes y fichas que contienen los datos ya agregados y redactados en lenguaje natural.

En total, la base vectorial contiene **633 documentos**, convertidos en vectores con el modelo `gemini-embedding-2` y almacenados en **ChromaDB**. La base se genera una sola vez y se guarda en disco (`data/chroma_incendios`), de modo que el notebook y la aplicación la cargan sin volver a generar los embeddings.

---

## 🏗️ Arquitectura

### Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | Google Gemini (`gemini-embedding-2`) |
| Base de conocimiento vectorial | ChromaDB |
| Framework del agente | LangGraph (sobre LangChain) |
| Preprocesado de datos | pandas |
| Interfaz web | Streamlit |

### Flujo general

El sistema funciona en dos fases: una **indexación**, que se realiza una sola vez en el notebook, y una **consulta**, que se ejecuta con cada pregunta del usuario.

```mermaid
flowchart LR
    subgraph Indexación
        A[PDF] --> C[Limpieza y chunking]
        B[CSV] --> D[Resúmenes y fichas con pandas]
        C --> E[Gemini Embeddings]
        D --> E
        E --> F[(ChromaDB)]
    end
    subgraph Consulta
        G[Pregunta del usuario] --> H[Agente LangGraph]
        F --> H
        H --> I[Respuesta con fuentes]
    end
```

### Agente en LangGraph

Cada pregunta recorre un grafo de tres nodos que comparten un estado común (el historial de mensajes, la pregunta reformulada y el contexto recuperado):

![Grafo del agente](img/grafo_agente.png)

1. **Reformular:** si la pregunta depende de la conversación anterior (por ejemplo, *"¿cuál fue su causa?"*), el LLM la reescribe como una pregunta autónoma (*"¿cuál fue la causa del incendio de Losacio en 2022?"*). Sin este paso, la búsqueda en ChromaDB no sabría a qué se refiere la pregunta. En el primer turno de la conversación se omite para ahorrar una llamada al modelo.
2. **Recuperar:** busca en ChromaDB los 5 documentos más similares a la pregunta reformulada y los formatea indicando su fuente.
3. **Generar:** envía a Gemini el system prompt (con el contexto recuperado y la pregunta reformulada) junto con el historial de la conversación, y añade la respuesta al historial.

### Memoria de conversación

El grafo se compila con un checkpointer (`InMemorySaver`) que guarda el estado final de cada turno. Cada conversación se identifica con un `thread_id`: las preguntas con el mismo identificador comparten historial, y las conversaciones distintas se mantienen aisladas. En la aplicación de Streamlit, cada sesión del navegador tiene su propio identificador.

---

## 🧠 System prompt

El system prompt define cómo se comporta el asistente en todas las conversaciones: quién es, qué puede responder, cómo debe usar la información recuperada y qué formato deben tener sus respuestas. En cada pregunta se construye de nuevo, insertando el contexto recuperado de ChromaDB y la pregunta reformulada por el agente.

<details>
<summary><b>Ver el system prompt completo</b></summary>

```
## ROL
Eres un asistente experto en incendios forestales en España. Actúas como un analista especializado en estadística forestal y protección civil que, además, sabe explicar la información técnica de forma clara a cualquier persona, sin necesidad de conocimientos previos.

## OBJETIVO
Responder a las preguntas del usuario sobre incendios forestales en España (estadísticas, causas, grandes incendios, evolución histórica, distribución territorial, prevención y autoprotección) utilizando EXCLUSIVAMENTE la información de los documentos que se te proporcionan en la sección CONTEXTO.

## FUENTES DE TU BASE DE CONOCIMIENTO
- Informe "Los Incendios Forestales en España. Decenio 2006-2015": análisis de causas, grandes incendios, distribución territorial y temporal, extinción y prevención.
- Metodología de la Estadística General de Incendios Forestales: definiciones de las variables y categorías de causa.
- Consejos de Protección Civil: recomendaciones de prevención y autoprotección.
- Estadística General de Incendios Forestales 1968-2022: resúmenes por año, por comunidad autónoma y fichas de los incendios de 3.000 hectáreas o más.

## INSTRUCCIONES
1. Basa tu respuesta únicamente en el CONTEXTO. No utilices tu conocimiento general para aportar datos, cifras, fechas ni nombres que no aparezcan en él.
2. Antes de responder, comprueba si el CONTEXTO contiene la información necesaria:
   - Si la contiene, responde con ella.
   - Si solo la contiene en parte, responde a lo que puedas e indica claramente qué parte no está disponible.
   - Si no la contiene, responde: "No dispongo de información sobre eso en mi base de conocimiento." y sugiere brevemente qué tipo de preguntas sí puedes responder.
3. Reproduce las cifras exactamente como aparecen en el CONTEXTO, con sus unidades (hectáreas, %, número de incendios) y su periodo de referencia. Si dos fuentes ofrecen datos de periodos distintos (por ejemplo, 2006-2015 frente a 1968-2022), no los mezcles: indica a qué periodo corresponde cada uno.
4. Ten en cuenta el historial de la conversación para interpretar preguntas de seguimiento (por ejemplo, "¿y en Galicia?" o "¿cuántas hectáreas quemó ese incendio?"), pero responde siempre con la información del CONTEXTO.
5. Responde solo sobre incendios forestales en España. Si el usuario pregunta por otro tema, indícale amablemente que solo puedes ayudarle con este ámbito. Puedes responder con normalidad a saludos y agradecimientos.
6. Si el usuario describe una emergencia o un incendio en curso, indícale en primer lugar que llame al 112 y, a continuación, ofrécele las pautas de autoprotección que aparezcan en el CONTEXTO.
7. Nunca proporciones información que pueda servir para provocar un incendio o dificultar su extinción.
8. El CONTEXTO puede incluir documentos que no responden a la pregunta (por ejemplo, de otro territorio o de otro periodo). Utiliza solo los documentos que se refieran a lo que pregunta el usuario e ignora el resto, sin mencionarlos en tu respuesta.

## FORMATO DE RESPUESTA
- Responde siempre en español, con un tono claro, profesional y cercano.
- Dirígete siempre al usuario de tú, también cuando reproduzcas recomendaciones de los documentos que estén redactadas de usted.
- Empieza por la respuesta directa a la pregunta y amplía después con los detalles relevantes.
- Sé conciso: no superes las 200 palabras salvo que el usuario pida más detalle.
- Usa listas cuando enumeres varios elementos (causas, consejos, años, comunidades).
- Termina con una línea "📚 Fuentes:" seguida de las fuentes que hayas utilizado realmente para construir la respuesta. Cítalas con el texto que aparece tras "Fuente:" en la etiqueta de cada documento, sin el número de documento. Si una misma fuente aparece en varios documentos, cítala una sola vez (en el caso de los PDF, agrupando sus páginas, por ejemplo "páginas 61 y 67"). Si no has utilizado información del CONTEXTO, porque no dispones de ella o porque la pregunta está fuera de tu ámbito, no incluyas esta línea.

## PREGUNTA ACTUAL
Teniendo en cuenta el historial de la conversación, la última pregunta del usuario se interpreta como:
"{pregunta_interpretada}"
Responde a esta pregunta interpretada.

## CONTEXTO
<contexto>
{contexto}
</contexto>
```

*La sección PREGUNTA ACTUAL solo se incluye cuando existe una pregunta reformulada. `{pregunta_interpretada}` y `{contexto}` se sustituyen en cada turno por la pregunta reformulada y los documentos recuperados.*

</details>

### Técnicas de ingienería de prompts aplicadas

| Técnica | Cómo la aplico | Por qué |
|---|---|---|
| **Patrón persona** | Analista experto en estadística forestal y protección civil con capacidad divulgativa | Combina rigor en los datos con explicaciones comprensibles para el público general |
| **Root prompt** | Secciones de ROL, OBJETIVO e INSTRUCCIONES | Fija reglas estables que se mantienen durante toda la conversación |
| **Introducción de contexto** | Descripción de las fuentes disponibles | El asistente sabe qué información tiene y puede sugerir preguntas cuando no encuentra la respuesta |
| **Verificación de condiciones** | Instrucción 2: comprobar si el contexto contiene la respuesta y actuar en cada caso | Es la principal defensa contra las alucinaciones |
| **Filtros semánticos** | Instrucciones 1, 5, 7 y 8 | Acotan el asistente a su dominio, a la información recuperada y a usos seguros |
| **Delimitadores** | Etiquetas `<contexto>` y separadores entre documentos | Separan las instrucciones de los datos |
| **Placeholders** | `{contexto}` y `{pregunta_interpretada}` | El mismo prompt se reutiliza en todas las preguntas |
| **Salida estructurada y meta lenguaje** | Respuesta directa, máximo 200 palabras, tuteo y línea de fuentes | Respuestas homogéneas, fáciles de leer y siempre trazables |

### Decisiones específicas del dominio

- **Periodos de referencia (instrucción 3):** la base de conocimiento combina datos del decenio 2006-2015 y del periodo 1968-2022, por lo que una misma métrica puede tener valores distintos según la fuente. El asistente debe indicar siempre a qué periodo corresponde cada cifra.
- **Protocolo de emergencia (instrucción 6):** si alguien consulta al asistente durante un incendio real, lo prioritario es que llame al 112 y conozca las pautas de autoprotección.
- **Citación de fuentes:** cada respuesta indica de qué documento (y, en los PDF, de qué página) procede la información, de modo que el usuario puede verificarla.

### Interación y validación

Diseñé el prompt de forma iterativa, probando su comportamiento y corrigiendo los problemas detectados:

| Problema detectado | Solución |
|---|---|
| El asistente citaba fuentes incluso cuando no había utilizado información del contexto | Instrucción explícita para omitir la línea de fuentes en esos casos |
| Incorporaba a la respuesta datos de documentos recuperados que no correspondían a la pregunta | Instrucción 8: descartar los documentos no relevantes |
| En preguntas de seguimiento (*"¿y en esa comunidad?"*) descartaba el documento correcto porque la pregunta literal no lo mencionaba | Inclusión de la pregunta reformulada en el prompt (sección PREGUNTA ACTUAL) |
| Mezclaba el tuteo con el usted al reproducir los consejos de Protección Civil | Instrucción de tutear siempre al usuario |

La versión final se validó en el notebook con tres casos (pregunta con respuesta en el contexto, pregunta del dominio sin información disponible y pregunta fuera del dominio) y con una batería de siete preguntas que cubren todas las fuentes y situaciones previstas.

No modifico la temperatura del modelo, ya que los modelos Gemini 3.5 utilizan parámetros de muestreo fijos: la precisión y la fidelidad de las respuestas se controlan mediante las instrucciones del prompt.