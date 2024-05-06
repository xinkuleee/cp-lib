class Solution:
	def calculate(self, s: str) -> int:
		sign = ['+', '-', '*', '/', '(', ')']
		v = []; num = ''
		for c in s:
			if c in sign:
				if num:
					v.append(num); num = ''
				if c == '-' and (not v or v[-1] == '('):
					v.append('0')
				v.append(c)
			elif c.isnumeric():
				num += c
		if num: v.append(num)

		stk0 = []; stk1 = []
		for e in v:
			if e.isnumeric():
				stk0.append(e)
			elif e in ['+', '-']:
				while stk1 and stk1[-1] in ['*', '/', '+', '-']:
					stk0.append(stk1.pop())
				stk1.append(e)
			elif e in ['*', '/', '(']:
				stk1.append(e)
			else:
				while stk1 and stk1[-1] != '(':
					stk0.append(stk1.pop())
				stk1.pop()
		while stk1:
			stk0.append(stk1.pop())

		res = []
		for e in stk0:
			if e.isnumeric():
				res.append(int(e))
			else:
				v = res.pop(); u = res.pop()
				if e == '+':
					res.append(u + v)
				if e == '-':
					res.append(u - v)
				if e == '*':
					res.append(u * v)
				if e == '/':
					res.append(u // v)
		return res[0]
