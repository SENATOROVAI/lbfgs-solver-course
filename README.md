# https://stepik.org/a/272346

2 Linear regression with the LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver method is a numerical optimization method used to find the minimum of an objective function. It is a gradient descent algorithm that uses an approximation of the Hessian matrix to minimize the function. 

Вот **реалистичный роадмап** по изучению алгоритма **L-BFGS-B** с нуля (2025–2026 годы). Предполагается, что у вас есть базовые знания математики вуза 1–2 курса и вы умеете программировать на Python.

| №  | Этап                              | Что изучить / понять глубоко                              | Время (при 8–15 ч/нед) | Лучшие ресурсы (2025–2026)                                                                 | Практика / чек-поинт                                      |
|----|-----------------------------------|-------------------------------------------------------------|--------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------|
| 1  | Базовая математика оптимизации   | Градиент, гессиан, квадратичная форма, выпуклость, градиентный спуск | 1–2 недели              | • 3Blue1Brown — Essence of calculus + Linear algebra<br>• Boyd & Vandenberghe — Convex Optimization (главы 2–3, 9) | Написать градиентный спуск на Rosenbrock, логировать путь |
| 2  | Градиентный спуск и его проблемы | Зигзаг, осцилляции, learning rate, momentum, Nesterov     | 1 неделя                | • Stanford CS231n лекция «Optimization» (YouTube)<br>• fast.ai Practical Deep Learning (часть про оптимизаторы) | Сравнить GD, Momentum, Nesterov на плохо обусловленной квадратичной функции |
| 3  | Метод Ньютона и его ограничения  | Классический Newton, Hessian, Cholesky, damping           | 1–1.5 недели            | • Nocedal & Wright «Numerical Optimization» — главы 3, 5, 6 (самые важные)<br>• YouTube: «Understanding BFGS» от aria42 | Реализовать damped Newton на 2–5-мерной задаче            |
| 4  | Quasi-Newton методы (общий взгляд) | Secant condition, DFP, BFGS формула обновления             | 1 неделя                | • Nocedal & Wright — глава 6 (6.1–6.3)<br>• Wikipedia → BFGS → Limited-memory BFGS       | Понять, почему BFGS лучше DFP и сохраняет положительную определённость |
| 5  | BFGS подробно                    | Две формы (B и H), two-loop recursion, Wolfe conditions   | 1.5–2 недели            | • Nocedal & Wright — 6.4–6.6, 8.2<br>• aria42 blog «Understanding L-BFGS» (2014, но всё ещё лучший)<br>• YouTube: «BFGS & L-BFGS: The Algorithms Behind Modern ML» | Реализовать классический BFGS (с матрицей Hₖ) на Rosenbrock |
| 6  | L-BFGS (Limited Memory)          | Two-loop recursion, H₀ = γI, хранение s и y векторов      | 2–3 недели              | • Nocedal & Wright — 7.2<br>• оригинальная статья Liu & Nocedal 1989<br>• Medium: «Implementing L-BFGS from scratch» Abhijit Mondal | Написать L-BFGS (unconstrained) — two-loop matvec без матриц |
| 7  | L-BFGS-B (box constraints)       | Projected gradient, Cauchy point, subspace minimization, active set | 2–4 недели              | • Оригинальная статья Zhu, Byrd, Nocedal 1997<br>• Fortran код на сайте Nocedal (для понимания логики)<br>• scipy.optimize._minimize_lbfgsb.py (читать!) | Добавить простые box constraints к своему L-BFGS → L-BFGS-B |
| 8  | Практика и отладка               | Line search (strong Wolfe), numerical stability, restart   | 2–4 недели              | • scipy.optimize код + тесты<br>• libLBFGS / L-BFGS++ / Optim.jl реализации | Сравнить свой L-BFGS-B с scipy на 5–10 тестовых функциях (CUTEst, etc.) |
| 9  | Современное применение и нюансы  | L-BFGS в PyTorch / JAX, Full-batch vs mini-batch, variance reduction, trust-region гибриды | 2–3 недели              | • PyTorch LBFGS (torch.optim)<br>• JAX — optax / jaxopt<br>• статьи 2020–2025 про L-BFGS в large-scale ML | Запустить L-BFGS на logistic regression / небольшой нейросети (full-batch) |
| 10 | Глубокое понимание               | Доказательства сходимости, суперлинейная скорость, preconditioning | +∞                      | • Nocedal & Wright полностью<br>• «Numerical Optimization» + статьи по trust-region / adaptive curvature | Написать объяснение L-BFGS-B на 5–7 страниц как будто для статьи |

### Рекомендуемый порядок ресурсов (самый эффективный путь 2025–2026)

1. 3Blue1Brown + CS231n Optimization lecture → быстрый старт
2. Nocedal & Wright «Numerical Optimization» — главы 2 → 3 → 5 → 6 → 7 → 8.2
3. aria42 blog «Understanding L-BFGS» + Medium Abhijit Mondal
4. Исходный код scipy.optimize L-BFGS-B → самый честный способ понять детали
5. Оригинальные статьи (Liu-Nocedal 1989 + Zhu-Byrd-Nocedal 1997)
6. Реализации на GitHub: smrfeld/l_bfgs_tutorial, L-BFGS++, jaxopt

### Примерный таймлайн (очень усреднённый)

- 0–6 недель  → уверенное понимание до BFGS
- 6–12 недель → L-BFGS (unconstrained) работает у вас с нуля
- 12–20 недель → L-BFGS-B (с box constraints) более-менее понятен и частично реализован
- 20–30+ недель → глубокое понимание + эксперименты в современных фреймворках

Удачи!  
Самое сложное — не код, а **интуиция**, почему two-loop recursion действительно приближает Hₖ без хранения матрицы. Именно на этом большинство людей «застревает». Поэтому рисуйте много картинок с эллипсами уровней модели и реальной функции.
