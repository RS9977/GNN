%{
#include <stdio.h>
%}

%%

"/*"            { /* ignore comments */ }
"*/"            { /* ignore comments */ }
"//".*          { /* ignore single line comments */ }

"int"           { printf("Keyword: %s\n", yytext); }
"return"        { printf("Keyword: %s\n", yytext); }

"("             { printf("Open Parenthesis\n"); }
")"             { printf("Close Parenthesis\n"); }
";"             { printf("Semicolon\n"); }
","             { printf("Comma\n"); }
"{"             { printf("Open Brace\n"); }
"}"             { printf("Close Brace\n"); }

[0-9]+          { printf("Number: %s\n", yytext); }
[a-zA-Z_][a-zA-Z0-9_]*  { printf("Identifier: %s\n", yytext); }

.               { /* ignore other characters */ }

%%

int main() {
   yylex();
   return 0;
}
