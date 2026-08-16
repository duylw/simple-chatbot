from fastapi import APIRouter, Depends, HTTPException, status, Request, Response

from src.schemas.agentic_ask import AskRequest, AgenticAskResponse
from src.dependencies import AgenticRAGDep, get_current_user
from src.models.user import User
from src.core.rate_limit import limiter

router = APIRouter(prefix="/agentic_ask", tags=["ask"])

@router.post("/", response_model=AgenticAskResponse)
@limiter.limit("10/minute")
async def ask_question(
    request: Request,
    response: Response,
    body: AskRequest,
    rag_service: AgenticRAGDep,
    current_user: User = Depends(get_current_user)
) -> AgenticAskResponse:
    try:
        results = await rag_service.ask(
            query=body.question,
            model=body.model,
            trace_user_id=str(current_user.id)
        )
        
        return AgenticAskResponse(
            query=body.question,
            rewritten_query=results.get("rewritten_query", ""),
            answer=results.get("answer", "No answer found."),
            sources=results.get("sources", []),
            n_iterations=results.get("n_iterations", 0),
            n_llm_calls=results.get("n_llm_calls", 0),
            execution_time=results.get("execution_time", 0.0),
            guardrail_result=results.get("guardrail_result")
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

