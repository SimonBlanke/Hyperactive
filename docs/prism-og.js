Prism.languages.og = {
	'comment': /\/\/.*/,
	'builtin': /\b(?:bool|byte|complex(?:64|128)|error|float(?:32|64)|rune|string|u?int(?:8|16|32|64)?|uintptr|append|cap|close|complex|copy|delete|imag|len|make|new|panic|print(?:ln)?|real|recover)\b/,
	'keyword': /\b(?:class|fn|in|break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go(?:to)?|if|import|interface|map|package|range|return|select|struct|switch|type|var)\b/,
	'boolean': /\b(?:_|iota|nil|true|false)\b/,
	'funcName': { // TODO
		pattern: /^(\w+) *(?=:?)(?=.+?)(?=->)/,
	},
	'returnType': {
		pattern: /(: )(\w+)/,
		lookbehind: true,
	},
	'funcCall': [
		{
			pattern: /(\w+)(?=\()/,
		},
		{
			pattern: /(\w+) *(?=->)/,
		},
		{
			pattern: /(\w+) *(?=: )/,
		},
		{
			pattern: /(\w+)(?=<)/,
		},
	],
	'secondary': {
		pattern: /(\.)(\w+)/,
		lookbehind: true
	},
	'assign': {
		pattern: /(\w+) +(?=:=)/,
	},
	'thisProp': {
		pattern: /(@)(\w+)/,
		lookbehind: true,
	},

	'externalReceiver': /(\w+)(?=::)/,
	'externalMethod': {
		pattern: /(::\*?)(\w+)/,
		lookbehind: true,
	},
	// 'lol': /::/,
	// 'externalMethod': {
	// 	pattern: /(\w+)(.)(\*?)(\w+)/g,
	// 	inside: {
	// 		externalReceiver: /(\w+)(?=::)/,
	// 		externalMethod: {
	// 			pattern: /(\*?)/,
	// 			lookbehind: true,
	// 		},
	// 	},
	// },
	'operator': /\{\}|,|\.|\[\]|[*\/%^!=]=?|\+[=+]?|-[=-]?|\|[=|]?|&(?:=|&|\^=?)?|>(?:>=?|=)?|<(?:<=?|=|-)?|:=|:|\.\.\./,
	'number': /(?:\b0x[a-f\d]+|(?:\b\d+\.?\d*|\B\.\d+)(?:e[-+]?\d+)?)i?/i,
	'this': /(@|this)/i,
	'argType': {
		pattern: /( +)(\w+)(?=(,|\)))/,
	},
	'structInst': {
		pattern: /(\w+)(?=\{)/,
	},
	'parens': /\(|\)/g,
	'string': {
		pattern: /(["'`])(\\[\s\S]|(?!\1)[^\\])*\1/,
		greedy: true
	}
};
// delete Prism.languages.og['class-name'];

Prism.languages.insertBefore('og', 'keyword', {
	'className': {
		pattern: /(struct|class|interface)(.*)/,
		lookbehind: true,
	},
	'packageName': {
		pattern: /^(\!)(\w*)/,
		lookbehind: true,
	},

});
