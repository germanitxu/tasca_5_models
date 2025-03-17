En aquesta tasca implementareu un generador de text basat en un GPT en miniatura. Així veurem com és el codi d'un transformer com el descrit als apunts.

Prendreu com a base l'exemple https://keras.io/examples/generative/text_generation_with_miniature_gpt/, i lliurareu el vostre quadern Colab (URL o document, com volgueu) amb les explicacions i modificacions que es demanen a continuació.

Realitzau les passes següents i responeu les qüestions que es plantegen aquí dins requadres de text dins el vostre quadern Colab.

1. Setup. Possiblement necessitareu instal·lar la darrera versió de keras. 

2. Implementació d'un bloc transformer com a capa. Què significa que la màscara d'atenció sigui causal?

3. Implementació de les capes d'embedding.

4. Implementació del GPT en miniatura. Quines funcions implementades als punts anteriors s'invoquen ara?

5. Dades per al model de llenguatge a nivell de paraula. Quina és l'aplicació habitual del dataset Imdb?

6. Implementació del callback Keras per generar text. Quin canvi faríeu al codi perquè sempre triàs la paraula més probable?

7. Entrenau el model amb un altre dataset de text d'una mida suficient.

8. Canviau el codi de generació de text, de forma que en lloc d'aturar quan ha generat un nombre de tokens, aturi quan genera un punt. D'aquesta forma, les frases generades sempre seran completes.

Xatbots basats en LLM

9. Comparau el rànquing de xatbots disponible a lmarena.ai amb el dels apunts. Quines diferències hi destacau?

10. Provau alguns LLM que hi ha disponibles a través de la interfície de xat de HuggingFace i comentau les diferències que hi heu observat. Hi ha models recents com DeepSeek 3 i Grok 3?