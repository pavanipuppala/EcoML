import React from 'react';

interface FormattedTextProps {
  text?: string | null;
  style?: React.CSSProperties;
  className?: string;
}

/**
 * Parses and renders text containing standard Markdown inline bold formatting (**text**)
 * using styled <strong> spans consistent with the dark theme.
 */
export const FormattedText: React.FC<FormattedTextProps> = ({ text, style, className }) => {
  if (!text) return null;

  // Split on **...** patterns while capturing the delimiter
  const parts = text.split(/(\*\*[^*]+\*\*)/g);

  return (
    <span style={style} className={className}>
      {parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
          const content = part.slice(2, -2);
          return (
            <strong key={idx} style={{ color: '#FFFFFF', fontWeight: 600 }}>
              {content}
            </strong>
          );
        }
        return part;
      })}
    </span>
  );
};

