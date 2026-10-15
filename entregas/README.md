# Entregas

Cada entrega es una carpeta con **dos ficheros**:

```
entregas/
  NOMBRE_DEL_EQUIPO/
    entrega_1/
      predicciones.csv
      mi_bot.py
    entrega_2/
      ...
```

- `NOMBRE_DEL_EQUIPO`: el nombre que os asigne la organización, **siempre el mismo**,
  en minúsculas, sin espacios ni tildes (por ejemplo `los_guardianes`).
- `entrega_1`, `entrega_2`… numeradas en orden. Sin límite de entregas.
  Última entrega: **19:00**.
- No modifiquéis carpetas de otros equipos ni entregas anteriores.

## 1. Generad el CSV (siempre con el script, nunca con Excel)

Antes del giro (15:30):
```
python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl --salida predicciones.csv
```

Desde las 15:30 (test + giro en un único CSV):
```
python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida predicciones.csv
```

## 2. Subidlo con un Pull Request

Necesitáis una cuenta de GitHub (basta con una por equipo).

**Desde la web, sin instalar nada:**
1. Pulsad **Fork** (arriba a la derecha) para tener vuestra copia del repositorio.
   Solo la primera vez.
2. En **vuestra copia**, entrad en `entregas/` → **Add file → Create new file**.
   Como nombre escribid la ruta completa, por ejemplo `los_guardianes/entrega_1/mi_bot.py`
   (al escribir `/` GitHub crea las carpetas), pegad vuestro código y pulsad
   **Commit changes**.
3. Ya dentro de la carpeta `entrega_1/` → **Add file → Upload files** → arrastrad
   `predicciones.csv` → **Commit changes**.
4. Volved a la portada de vuestra copia → **Contribute → Open pull request**.
   Título: `NOMBRE_DEL_EQUIPO entrega N`. Pulsad **Create pull request**.

Para la siguiente entrega: en vuestra copia pulsad antes **Sync fork → Update branch**
y repetid desde el paso 2 con `entrega_2`.

**Con git:** igual, pero desde vuestro fork clonado: añadid la carpeta, *commit*,
*push* y abrid el *Pull Request* hacia `main`.

La organización puntúa cada Pull Request y os comunica la **nota pública**. El
leaderboard en directo está proyectado en la sala.

> Todo lo que se sube aquí es visible para todos. Las reglas prohíben copiar el
> código de otro equipo, y el jurado revisa el código de cada entrega.
