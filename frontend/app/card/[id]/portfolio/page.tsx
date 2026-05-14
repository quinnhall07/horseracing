import { redirect } from "next/navigation";

interface PageProps {
  params: { id: string };
}

/**
 * The old /portfolio sub-route was a dedicated bet-ticket page. The
 * pareto-driven rebuild merged it into the parent /card/[id] view (Stream Z).
 * Anyone bookmarking the old URL gets redirected to the unified result page,
 * anchored to the bet ticket area.
 */
export default function PortfolioRedirect({ params }: PageProps) {
  redirect(`/card/${params.id}#races`);
}
