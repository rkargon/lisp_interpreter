;; bools
(set if (lambda (condition if_true if_false) (cond (condition if_true) (true if_false))))
(set and (lambda (a b) (if a b a)))
(set or (lambda (a b) (if a a b)))
(set not (lambda (a) (if a false true)))

;; func ops
(set comp (lambda (f g) (lambda (x) (f (g x)))))
(set fog comp)
(set let (macro (name, value, body) (list (list lambda (list name) body) value)))
(set letrec (macro (name, value, body) (list (list lambda () (list (list lambda (list name) body) (list set name value))))))
(set Y (lambda (f) ((lambda (x) (lambda (n) (f (x x) n))) (lambda (x) (lambda (n) (f (x x) n))))))

;; misc
(set dec (lambda (n) (- n 1)))
(set inc (lambda (n) (+ n 1)))
(set square (lambda (n) (* n n)))

;; list ops
(set is_nil (lambda (l) (= l nil)))
(set fold (lambda (f l i) (if (is_nil l) i   (f (head l) (fold f (tail l) i)))))
(set map (lambda (f l)   (if (is_nil l) nil (cons (f (head l)) (map f (tail l))))))
(set map_val (lambda (i l) (map (lambda (_) i) l)))
(set filter (lambda (f l) (if (is_nil l) nil (let rest (filter f (tail l)) (if (f (head l)) (cons (head l) rest) rest)))))
(set len (lambda (l) (fold add (map_val 1 l) 0)))
(set cat (lambda (l1 l2) (if (is_nil l1) l2 (cons (head l1) (cat (tail l1) l2)))))
(set range (lambda (start end) (if (>= start end) nil (cons start (range (+ start 1) end)))))
(set flatten (lambda (l) (fold cat l nil)))
(set cartesian_prod (lambda (l1 l2) (flatten (map (lambda (a) (map (lambda (b) (list a b)) l2)) l1))))
(set first head)
(set second (comp head tail))
(set zip (lambda (l1 l2) (if (is_nil l1) nil (cons (list (head l1) (head l2)) (zip (tail l1) (tail l2))))))

;; iterators
;; # an iterator is defined as a pair where the first element is the current value, and the second is a lambda that when called yields the next iterator (i.e. the next pair of (value, lambda)
;; # i.e. (list current_value (lambda () (next_iterator...)))
(set icount (lambda (n) (list n (lambda () (icount (inc n))))))
(set imap (lambda (f i) (list (f (head i)) (lambda () (imap f (inext i))))))
(set inext (lambda (i) ((second i))))
(set itake (lambda (n i) (if (= 0 n) nil (cons (head i) (itake (dec n) (inext i))))))
(set ifilter (lambda (f i) (let rest (ifilter f (inext i)) (if (f (head i)) (list (head i) (lambda () rest)) rest))))
(set ifilter_mod (lambda (n i) (ifilter (lambda (m) (!= 0 (% n m))) i)))
(set sieve (lambda (i) (list (head i) (lambda () (sieve (ifilter_mod (head i) i))))))
(set ienumerate (lambda (i) (izip (icount 0) i)))
(set izip (lambda (i1 i2) (list (list (head i1) (head i2)) (lambda () (izip (inext i1) (inext i2))))))


;; macro shit
(set letrec2 (macro (name value body) (or (assert (in (head value) (list lambda macro)) "value must be lambda or macro!") (letlist (l_ l_params l_body) value (list (quote let) name (list Y (list l_ (cons name l_params) l_body)) body)))))
;;#  TODO may this work with (letlist (x y z) values_list (body...)))
;; maybe do (letlist (x v) (y v2) ... body)?
(set letlist (macro (names values body) (fold (lambda (nv_pair, rest) (list (quote let) (first nv_pair) (second nv_pair) rest)) (zip names values) body)))
