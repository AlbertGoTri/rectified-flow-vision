# Rectified Flow: Explicación Técnica

## ¿Qué es Flow Matching?

Flow Matching es una técnica para entrenar modelos generativos que aprenden a transformar una distribución simple (ruido gaussiano) en una distribución compleja (imágenes reales).

### La idea básica

Imagina que quieres ir del punto A (ruido) al punto B (imagen). Flow Matching aprende el **campo de velocidad** $v(x_t, t)$ que te dice en cada momento hacia dónde moverte:

$$\frac{dx_t}{dt} = v(x_t, t)$$

Con interpolación lineal, definimos:
$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

Donde:
- $x_0$ = ruido gaussiano $\sim \mathcal{N}(0, I)$
- $x_1$ = imagen real
- $t \in [0, 1]$ = tiempo

El campo de velocidad óptimo es simplemente: $v^* = x_1 - x_0$

## El Problema: Trayectorias Curvas

Cuando entrenas un modelo flow estándar, las trayectorias aprendidas suelen ser **curvas** porque:

1. El modelo promedia sobre muchos pares $(x_0, x_1)$ diferentes
2. La "dirección promedio" desde cualquier punto intermedio no es recta

Esto significa que necesitas **muchos pasos** de integración (Euler) para seguir la curva correctamente. Típicamente 100-1000 pasos.

```
Ruido → → → → → → → → → → Imagen
      ↘               ↗
        ↘           ↗
          ↘       ↗
            ↘   ↗
              ↓
        (trayectoria curva)
```

## La Solución: Rectified Flow (Reflow)

La idea clave de Reflow es **enderezar** las trayectorias iterativamente:

### Paso 1: Entrenar modelo base
Entrena un flow model normal con datos reales.

### Paso 2: Generar pares "acoplados"
Usa el modelo base para generar pares $(z, x)$:
- $z$ = ruido inicial (lo guardas)
- $x$ = imagen generada (la guardas)

### Paso 3: Entrenar modelo rectificado
Entrena un NUEVO modelo usando estos pares acoplados.

La magia: como cada $z$ específico siempre va a su $x$ específico, el modelo aprende trayectorias más directas.

```
Después de Reflow:

Ruido → → → → → → → → → → Imagen
                          ↑
        (trayectoria recta!)
```

### Iteraciones múltiples (Reflow-K)

Puedes aplicar Reflow múltiples veces:
- **Reflow-1**: Una iteración (mejora significativa)
- **Reflow-2**: Dos iteraciones (aún más recto)
- **Reflow-K**: K iteraciones

Con cada iteración, las trayectorias son más rectas y necesitas menos pasos.

## Beneficios Cuantitativos

| Modelo | Pasos necesarios | Calidad |
|--------|------------------|---------|
| Base   | 100-1000         | Buena   |
| Reflow-1 | 10-50          | Similar |
| Reflow-2 | 4-10           | Similar |
| Reflow-3 | 1-4            | Similar |

¡Potencialmente 100x más rápido!

## Intuición Matemática

Sea $\pi_t$ la distribución marginal en tiempo $t$.

- **Modelo base**: Aprende $E[v | x_t]$ promediando sobre todos los posibles $(x_0, x_1)$ que pasan por $x_t$
- **Modelo Reflow**: Los pares están "acoplados" - cada $x_0$ tiene un único $x_1$ correspondiente

Este acoplamiento reduce la varianza del target y produce trayectorias más deterministas.

## Conexión con Otras Técnicas

- **Consistency Models** (OpenAI): Idea similar de "saltar" pasos
- **Progressive Distillation**: Reducir pasos gradualmente
- **DDIM**: Sampling determinista en diffusion models

Rectified Flow es elegante porque:
1. No requiere cambiar la arquitectura
2. El proceso es simple e interpretable
3. Se puede aplicar iterativamente

## En Este Código

```python
# 1. Modelo base - aprende v(x_t, t) con pares aleatorios
base_model = BaseFlowModel(...)
train_base_flow(base_model, dataloader)

# 2. Generar pares acoplados
x0_data, x1_data = generate_reflow_pairs(base_model, num_pairs=10000)

# 3. Modelo rectificado - aprende con pares acoplados
rect_model = RectifiedFlowModel(...)
train_rectified_flow(rect_model, x0_data, x1_data)

# 4. Comparar velocidad
# Base: 100 pasos necesarios
# Rectificado: 4 pasos suficientes
```

## Referencias

1. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**
   - Liu et al., 2022
   - https://arxiv.org/abs/2209.03003

2. **Flow Matching for Generative Modeling**
   - Lipman et al., 2022
   - https://arxiv.org/abs/2210.02747

3. **Building Normalizing Flows with Stochastic Interpolants**
   - Albergo & Vanden-Eijnden, 2022
   - https://arxiv.org/abs/2209.15571
