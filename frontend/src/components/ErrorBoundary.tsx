import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  componentStack: string;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, componentStack: '' };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error, componentStack: '' };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ componentStack: errorInfo.componentStack ?? '' });
    console.error('User detail subtree crashed:', error);
    console.error(errorInfo.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="space-y-3">
          {this.props.fallback ?? (
            <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
              <p className="font-medium">頁面載入失敗</p>
              <p className="mt-2 text-xs break-words text-red-500">
                請稍後再試一次。
              </p>
            </div>
          )}
          <details className="rounded-xl border border-red-200 bg-white px-5 py-4 text-sm text-red-600">
            <summary className="cursor-pointer font-medium">除錯資訊</summary>
            <div className="mt-3 space-y-2">
              <p className="break-words text-xs text-red-500">
                {this.state.error?.message ?? 'Unknown error'}
              </p>
              {this.state.componentStack && (
                <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-red-50 p-3 text-[11px] leading-5 text-red-500">
                  {this.state.componentStack}
                </pre>
              )}
            </div>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
