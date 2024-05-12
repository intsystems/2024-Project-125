% This style file needs LaTeX version 2e
\NeedsTeXFormat{LaTeX2e}

% Style name
\ProvidesPackage{icorsdssv2024}[2024/01/26]

\LoadClass[12pt,letterpaper]{article}

% Packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T2A]{fontenc}
\RequirePackage{amsmath, amsfonts, amssymb, bm}
\RequirePackage{graphicx}
%\RequirePackage[letterpaper, left=2cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\RequirePackage[letterpaper, left=3cm,right=3cm,top=3cm,bottom=3cm]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage[makeindex]{imakeidx}

\makeatletter

% Running title and authors, address
\newcommand{\mytitle}[1]{\def\@mytitle{#1}}
\newcommand{\mytitlerunning}[1]{\def\@mytitlerunning{#1}}
\newcommand{\myauthor}[1]{\def\@myauthor{#1}}
\newcommand{\myauthorrunning}[1]{\def\@myauthorrunning{#1}}
\newcommand{\myaddress}[1]{\def\@myaddress{#1}}
\newcommand{\speaker}[1]{\textbf{#1}}

% Headings and foots
\setlength{\headheight}{15pt}
\AtBeginDocument{\renewcommand{\headrulewidth}{0.5pt}}
\pagestyle{fancy}
%%%\fancypagestyle{plain}
\AtBeginDocument{\lhead{\textit{\@myauthorrunning}}\chead{}\rhead{\textit{\@mytitlerunning}}}
\AtBeginDocument{\renewcommand{\headrulewidth}{0.5pt}}
\AtBeginDocument{\renewcommand{\footrulewidth}{0.5pt}}
\lfoot{\textbf{MIPT Innovation Workshop Conference 2024}}\cfoot{}\rfoot{\thepage}

% Abstract and keywords
\def\@abstract{\normalsize\textbf{Abstract. }} \def\end@abstract{\par}
\def\@mykeywords{\normalsize\textbf{Keywords. }} \def\end@mykeywords{\par}
\newtoks\abstract
\newtoks\mykeywords
\def\cleararticleinfo{\global\abstract={}}
\def\cleararticleinfo{\global\mykeywords={}}

% Make title
\newcommand{\mymaketitle}{
\newsavebox{\foo}
\savebox{\foo}{
\begin{minipage}[t]{14.9cm}
	% Title and authors
	\begin{minipage}[c][4cm][c]{14.7cm}
%\begin{minipage}[t]{17.3cm}
%	% Title and authors
%	\begin{minipage}[c][4cm][c]{17.1cm}
		\begin{center}
		\Large\bfseries\@mytitle
		\par\vspace{1em}
		\normalfont\normalsize\noindent\@myauthor
		\vspace{-0.5em}
		\end{center}
	\end{minipage}
	% Affiliations
	\par\vspace{1em}\itshape\small
	\@myaddress
	\par\vspace{1em}\noindent\rule{\linewidth}{.5pt}
	% Abstract
	\normalfont
	\par\vspace{1em}
	\begin{@abstract}\the\abstract\end{@abstract}
	% Keywords
        \par\vspace{1em}
	\itshape
	\begin{@mykeywords}\the\mykeywords\end{@mykeywords}
	
	
	\normalfont
	\par\vspace{1em}
	\normalsize{UDC:} 519.246 + 519.837
	\par\noindent\rule{\textwidth}{.5pt}
\end{minipage}}
\par\noindent\usebox{\foo}\vspace{-0.25cm}}

\setlength{\parskip}{\baselineskip}

% Do not indent the 1st line of a paragraph.
\setlength{\parindent}{0mm}

% For making the biblipgraphy more compact
%\renewcommand{\itemsep}{\skip0}
%\renewcommand{\parsep}{\skip0}

%\makeatother



\endinput
