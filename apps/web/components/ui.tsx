"use client";

import React from "react";

export function cx(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}

export function Badge({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: "neutral" | "good" | "warn" | "bad";
}) {
  const cls =
    variant === "good"
      ? "bg-emerald-100 text-emerald-800 border-emerald-200"
      : variant === "bad"
      ? "bg-rose-100 text-rose-800 border-rose-200"
      : variant === "warn"
      ? "bg-amber-100 text-amber-900 border-amber-200"
      : "bg-zinc-100 text-zinc-800 border-zinc-200";

  return (
    <span className={cx("inline-flex items-center border rounded-full px-3 py-1 text-xs font-semibold", cls)}>
      {children}
    </span>
  );
}

export function Card({
  title,
  subtitle,
  right,
  children,
}: {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white border rounded-2xl shadow-sm">
      {(title || subtitle || right) && (
        <div className="p-5 border-b flex items-start justify-between gap-3 flex-wrap">
          <div>
            {title && <div className="text-sm font-semibold text-zinc-900">{title}</div>}
            {subtitle && <div className="text-xs text-zinc-500 mt-1">{subtitle}</div>}
          </div>
          {right}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  title,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "outline" | "ghost";
  title?: string;
  type?: "button" | "submit";
}) {
  const base = "px-4 py-2 rounded-lg text-sm font-medium transition";
  const cls =
    variant === "primary"
      ? disabled
        ? "bg-zinc-200 text-zinc-600 cursor-not-allowed"
        : "bg-zinc-900 text-white hover:bg-zinc-800"
      : variant === "outline"
      ? disabled
        ? "border text-zinc-400 cursor-not-allowed"
        : "border hover:bg-zinc-50"
      : disabled
      ? "text-zinc-400 cursor-not-allowed"
      : "hover:bg-zinc-100";

  return (
    <button type={type} className={cx(base, cls)} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

export function SmallLink({
  href,
  children,
  newTab = false,
}: {
  href: string;
  children: React.ReactNode;
  newTab?: boolean;
}) {
  return (
    <a
      className="text-xs px-2 py-1 rounded-lg border hover:bg-zinc-50 inline-flex items-center gap-1"
      href={href}
      target={newTab ? "_blank" : undefined}
      rel={newTab ? "noreferrer" : undefined}
    >
      {children}
    </a>
  );
}

export function Divider() {
  return <div className="h-px bg-zinc-100 my-4" />;
}
