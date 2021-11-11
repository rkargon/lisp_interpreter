(set true (lambda (x y) x))
(set false (lambda (x y) y))
(set if (lambda (cond t f) (cond t f)))
(set and (lambda (a b) (if a b a)))
(set or (lambda (a b) (if a a b)))
(set not (lambda (a) (if a false true)))

(set pair (lambda (x y) (lambda (z) (z x y))))
(set first (lambda (l) (l (lambda (x y) x))))
(set second (lambda (l) (l (lambda (x y) y))))

;; LISTS

;; INTEGERS
(set zero  (lambda (f) (lambda (x) x)))
(set one   (lambda (f) (lambda (x) (f x))))
(set two   (lambda (f) (lambda (x) (f (f x)))))
(set three (lambda (f) (lambda (x) (f (f (f x))))))
(set four  (lambda (f) (lambda (x) (f (f (f (f x)))))))
(set succ (lambda (n) (lambda (f) (lambda (x) (f ((n f) x))))))
(set add (lambda (n m) (lambda (f) (lambda (x) ((m f) ((n f) x))))))
(set mul (lambda (n m) (lambda (f) (lambda (x) ((m (n f)) x)))))
(set mul2 (lambda (n m) (lambda (f) (m (n f)))))
(set exp (lambda (n m) (lambda (f) (lambda (x) (m n)))))
(set church (lambda (n) (if (= n 0) zero (succ (church (- n 1))))))
(set is_zero (lambda (n) ((n (lambda (x) false)) true)))


