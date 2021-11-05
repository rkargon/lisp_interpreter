;; bools
(set if (lambda (condition if_true if_false) (cond (condition if_true) (true if_false))))
(set and (lambda (a b) (if a b a)))
(set or (lambda (a b) (if a a b)))
(set not (lambda (a) (if a false true)))

;; func ops
(set comp (lambda (f g) (lambda (x) (f (g x)))))
(set fog comp)

;; list ops
(set is_nil (lambda (l) (= l nil)))
(set fold (lambda (f l i) (if (is_nil l) i   (f (head l) (fold f (tail l) i)))))
(set map (lambda (f l)   (if (is_nil l) nil (cons (f (head l)) (map f (tail l))))))
(set map_val (lambda (i l) (map (lambda (_) i) l)))
(set len (lambda (l) (fold add (map_val 1 l) 0)))
(set cat (lambda (l1 l2) (if (is_nil l1) l2 (cons (head l1) (cat (tail l1) l2)))))
(set range (lambda (start end) (and (assert (<= start end) (print (list start "<=" end))) (if (= start end) nil (cons start (range (+ start 1) end))))))
(set flatten (lambda (l) (fold cat l nil)))
(set cartesian_prod (lambda (l1 l2) (flatten (map (lambda (a) (map (lambda (b) (list a b)) l2)) l1))))
(set first head)
(set second (comp head tail))
