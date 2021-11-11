(set true (lambda (x y) x))
(set false (lambda (x y) y))
(set if (lambda (cond t f) (cond t f)))

(set pair (lambda (x y) (lambda (z) (z x y))))
(set first (lambda (l) (l (lambda (x y) x))))
(set second (lambda (l) (l (lambda (x y) y))))
